# -*- coding: utf-8 -*-
# ============================================================
# ALS (implicit) + BM25 + recency + cooc-rerank — Coarse→Refine FAST SWEEP
# + LGBM RERANKER (user-aware + trend + financial + user-cluster)
#
# Noutăți față de versiunea anterioară:
#  - Validare temporală + subsampling pentru Stage-1 (coarse)
#  - Feature-uri RECENCY & TREND (10m / 1h / 24h), pool_len & pool_len_delta,
#    days_since_anchor_seen
#  - Features user-aware (brand/cat share + recency) + FIN din CSV (fail-safe)
#  - Hard negatives: blend cosine + co-oc (lift/npmi/jacc) pentru "greutate"
#  - Group weights by recency (mai recent => greutate mai mare) în LambdaRank
#  - LightGBM: constrângeri MONOTONE pe features "cu semn clar"
#  - HPO cu Optuna (maximize nDCG@5) + early stopping
#  - Ensembling multi-seed (3x) și, opțional, pe perioade
#  - Clustering ușor pe useri (MiniBatchKMeans) -> user_cluster_id (opțional)
#  - Cache-uri și vectorizare unde a contat + MMR robust
# ============================================================

import os, glob, json as _json, math, pickle, random, warnings, time, hashlib, re
warnings.filterwarnings("ignore")
from collections import Counter, defaultdict
from itertools import combinations, product
from datetime import datetime, timezone

import numpy as np
np.seterr(all="ignore")
import pandas as pd
import scipy.sparse as sp

import implicit
from implicit.nearest_neighbours import bm25_weight
import lightgbm as lgb

from datetime import datetime, timezone

# ---------------- Config "knobs" ----------------
ORDERS_PATH = None            # auto-detect orders_*.json dacă e None
USE_ONLY_COMPLETED = True
SEED = 42
TEST_SPLIT = 0.20
MIN_BASKET_LEN = 2
MIN_USER_EVENTS = 3
MIN_ITEM_EVENTS = 3

# Sweep + runtime
FAST_SWEEP = True
MAX_TRIALS = 250
EVAL_SAMPLE_FRAC = 0.40
ALS_IT_CAP = 18
TOPN_REFINE = 8

# Reranker LGBM
TRAIN_LGBM_RERANKER = True
USE_OPTUNA = True
ENSEMBLE_SEEDS = [42, 777, 1313]  # ensembling pe seed-uri
LGBM_MODEL_PATH = "lgbm_reranker.txt"

# Fișier financiar (opțional)
FINANCIAL_PATH = "financial_data.csv"

# Artefacte / logging
RESULTS_CSV   = "sweep_results.csv"
RESULTS_JSONL = "sweep_results.jsonl"
BEST_PKL      = "best_model.pkl"
TEMP_PKL      = "best_model_ckpt.pkl"
SAVE_INTERMEDIATE_BEST_EVERY = 10

# Evaluare
EVAL_TOPKS = (5, 10)
PRIMARY_K = 5
SCORE_KEY = ("MAP@K","Precision@K","Recall@K")

# ALS/BM25 grids
HALF_LIFE_GRID = [15., 22.5, 30., 45., 60., 75., 90., 105., 120., 150., 180., 210., 240., 300.]
BM25_GRID = [(1.0,0.55), (1.0,0.65), (1.0,0.75), (1.0,0.85),
             (1.2,0.75), (1.4,0.75), (1.6,0.75), (1.4,0.85)]
ALS_GRID = [
    {"K":160, "IT":35, "REG":0.08, "ALPHA":55.0},
    {"K":224, "IT":40, "REG":0.10, "ALPHA":60.0},
    {"K":256, "IT":45, "REG":0.11, "ALPHA":60.0},
    {"K":320, "IT":45, "REG":0.12, "ALPHA":65.0},
    {"K":384, "IT":50, "REG":0.13, "ALPHA":70.0},
    {"K":448, "IT":55, "REG":0.14, "ALPHA":75.0},
    {"K":512, "IT":55, "REG":0.16, "ALPHA":80.0},
    {"K":576, "IT":60, "REG":0.18, "ALPHA":85.0},
]

# 5 profile de rerank/MMR/pool/long-tail (îți poți adăuga altele)
RERANK_PROFILES = [
    dict(TAU=1.00, ALPHA=0.75, BETA=0.14, GAMMA=0.08,  DELTA=0.02,
         POP_EPS=0.008, LAMBDA_SHRINK=40.0, CAND_POOL=200,
         MMR_ENABLE=True,  MMR_LAMBDA=0.95, MMR_MIN_RATIO=2.0,
         RERANK_ONLY_LONGTAIL=True,  POP_PCTL=60, POOL_BASE=35, POOL_EXTRA=65),
    dict(TAU=1.00, ALPHA=0.72, BETA=0.16, GAMMA=0.08,  DELTA=0.02,
         POP_EPS=0.010, LAMBDA_SHRINK=35.0, CAND_POOL=280,
         MMR_ENABLE=True,  MMR_LAMBDA=0.97, MMR_MIN_RATIO=1.8,
         RERANK_ONLY_LONGTAIL=True,  POP_PCTL=70, POOL_BASE=55, POOL_EXTRA=120),
    dict(TAU=0.90, ALPHA=0.78, BETA=0.12, GAMMA=0.07,  DELTA=0.03,
         POP_EPS=0.006, LAMBDA_SHRINK=45.0, CAND_POOL=160,
         MMR_ENABLE=True,  MMR_LAMBDA=0.93, MMR_MIN_RATIO=2.5,
         RERANK_ONLY_LONGTAIL=True,  POP_PCTL=50, POOL_BASE=25, POOL_EXTRA=40),
    dict(TAU=1.05, ALPHA=0.73, BETA=0.18, GAMMA=0.06,  DELTA=0.03,
         POP_EPS=-0.006, LAMBDA_SHRINK=38.0, CAND_POOL=240,
         MMR_ENABLE=True,  MMR_LAMBDA=0.94, MMR_MIN_RATIO=2.0,
         RERANK_ONLY_LONGTAIL=True,  POP_PCTL=65, POOL_BASE=45, POOL_EXTRA=100),
    dict(TAU=0.95, ALPHA=0.74, BETA=0.14, GAMMA=0.08,  DELTA=0.04,
         POP_EPS=0.004, LAMBDA_SHRINK=42.0, CAND_POOL=220,
         MMR_ENABLE=True,  MMR_LAMBDA=0.96, MMR_MIN_RATIO=2.2,
         RERANK_ONLY_LONGTAIL=False, POP_PCTL=55, POOL_BASE=35, POOL_EXTRA=80),
]

TOP_NEIGHBORS = 40
NUM_THREADS = max(1, os.cpu_count() or 4)
USE_GPU = bool(int(os.getenv("ALS_GPU", "0")))  # export ALS_GPU=1 dacă ai CUDA

random.seed(SEED); np.random.seed(SEED)

# ---------------- IO helpers ----------------
def find_orders_path():
    cands = sorted(glob.glob("*.json"), key=os.path.getmtime, reverse=True)
    for p in cands:
        if "orders" in p.lower(): return p
    if os.path.exists("orders_generated_xl.json"): return "orders_generated_xl.json"
    raise FileNotFoundError("Nu găsesc fișierul orders_*.json.")

ORDERS_PATH = ORDERS_PATH or find_orders_path()
print(f"[INFO] Folosesc comenzi din: {ORDERS_PATH}")

def parse_dt(s):
    """
    Parsează ISO 8601 și întoarce mereu un datetime AWARE (UTC).
    Acceptă 'Z' la final și șiruri fără TZ (considerate UTC).
    """
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def safe_load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        try:
            return _json.load(f)
        except _json.JSONDecodeError as e:
            f.seek(0); raw = f.read()
            cutoff = raw[:e.pos]
            last_valid = max(cutoff.rfind('}'), cutoff.rfind(']'))
            if last_valid != -1:
                try: return _json.loads(f"[{cutoff[:last_valid+1]}]")
                except: return []
            return []

orders = safe_load_json(ORDERS_PATH)
print(f"[INFO] Loaded orders: {len(orders)}")

# ---------------- Split (temporal) ----------------
def user_temporal_split(orders, test_ratio=0.2):
    by_user = {}
    for o in orders:
        u = o.get("clientId"); dt = parse_dt(o.get("orderDate"))
        if u is None or dt is None: continue
        if USE_ONLY_COMPLETED and o.get("status") != "COMPLETED": continue
        by_user.setdefault(u, []).append(o)
    train, test = [], []
    for _, lst in by_user.items():
        lst.sort(key=lambda x: parse_dt(x["orderDate"]))
        cut = max(1, int(len(lst) * (1 - test_ratio)))
        train += lst[:cut]; test += lst[cut:]
    train.sort(key=lambda x: parse_dt(x["orderDate"]))
    test.sort(key=lambda x: parse_dt(x["orderDate"]))
    return train, test

train_orders, test_orders = user_temporal_split(orders, TEST_SPLIT)
print(f"[INFO] Train: {len(train_orders)} | Test: {len(test_orders)} (leave-last per user)")

# ---------------- Co-oc pe TRAIN + META user/item ----------------
item_count, pair_count = Counter(), Counter()
train_baskets = []
for o in train_orders:
    s = {op.get("productId") for op in o.get("orderProducts", []) if op.get("productId") is not None}
    if len(s) >= 2:
        train_baskets.append(s)
        for i in s: item_count[i] += 1
        for a,b in combinations(sorted(s),2): pair_count[(a,b)] += 1
N_train_baskets = max(1, len(train_baskets))
MAX_ITEM_COUNT = max(item_count.values()) if item_count else 1

# ---------------- Bought-Together rules (association) ----------------
# Extrage nume produse (dacă există în JSON) pentru un output mai lizibil
ITEM_NAME = {}
def _maybe_update_name(op):
    pid = op.get("productId")
    if pid is None:
        return
    for k in ("productName", "name", "title", "skuName"):
        val = op.get(k)
        if val:
            ITEM_NAME.setdefault(pid, str(val))
            break

for o in train_orders:
    for op in (o.get("orderProducts") or []):
        _maybe_update_name(op)

def _p(x):
    return x / float(N_train_baskets if N_train_baskets else 1)

def format_rule(a, b, supp, ca, cb):
    conf_ab = supp / float(ca if ca else 1)
    pa, pb, pab = _p(ca), _p(cb), _p(supp)
    lift_ab = pab / (pa * pb + 1e-12)
    return {
        "A": int(a), "B": int(b),
        "conf": conf_ab, "support": int(supp),
        "lift": float(lift_ab),
        "countA": int(ca), "countB": int(cb),
        "A_name": ITEM_NAME.get(a, str(a)),
        "B_name": ITEM_NAME.get(b, str(b)),
    }

def bought_together_rules(min_support=10, min_conf=0.08, min_lift=1.05, top=20):
    rules = []
    for (a, b), supp in pair_count.items():
        ca, cb = item_count.get(a, 0), item_count.get(b, 0)
        if supp < min_support:
            continue
        if ca > 0:
            r = format_rule(a, b, supp, ca, cb)
            if (r["conf"] >= min_conf) and (r["lift"] > min_lift):
                rules.append(r)
        if cb > 0:
            r = format_rule(b, a, supp, cb, ca)
            if (r["conf"] >= min_conf) and (r["lift"] > min_lift):
                rules.append(r)
    rules.sort(key=lambda r: (r["conf"], r["lift"], r["support"]), reverse=True)
    return rules[:top]

def print_bt_summary(rules):
    print("[BT] Rezumat reguli A→B (confidence = P(B|A)) — METRICĂ DE REGULĂ, nu acuratețe de sistem.")
    if not rules:
        print("[BT] Nicio regulă care să treacă pragurile.")
        return
    print(f"[BT] Afișez top {len(rules)} reguli după confidence:")
    for i, r in enumerate(rules, start=1):
        print(f"{i:3d}. {r['A']} → {r['B']} | conf={r['conf']*100:6.2f}% | support={r['support']:4d} | "
              f"lift={r['lift']:.3f} | count(A)={r['countA']:4d} | count(B)={r['countB']:4d} | "
              f"A_name='{r['A_name']}' | B_name='{r['B_name']}'")

def percentile_pop(pctl: int):
    return float(np.percentile(list(item_count.values()), pctl)) if item_count else 0.0

ITEM_META = {}
USER_BRAND_CNT = Counter()
USER_CAT_CNT   = Counter()
USER_TOTAL     = Counter()
USER_LAST_DATE = {}

def _safe_str(x): return str(x) if x is not None else "unknown"

for o in train_orders:
    uid = o.get("clientId"); odt = parse_dt(o.get("orderDate"))
    if uid is None or odt is None: continue
    # odt este AWARE din parse_dt; fallback defensiv (nu ar trebui să intre):
    if odt.tzinfo is None:
        odt = odt.replace(tzinfo=timezone.utc)
    USER_LAST_DATE[uid] = max(USER_LAST_DATE.get(uid, odt), odt)
    seen = set()
    for op in (o.get("orderProducts") or []):
        pid = op.get("productId")
        if pid is None or pid in seen: continue
        seen.add(pid)
        brand = _safe_str(op.get("brandName"))
        cat   = _safe_str(op.get("category"))
        ITEM_META.setdefault(pid, {"brand": brand, "category": cat})
        USER_BRAND_CNT[(uid, brand)] += 1
        USER_CAT_CNT[(uid, cat)]     += 1
        USER_TOTAL[uid]              += 1

def _pop(pid): return math.log1p(item_count.get(pid, 0))
def _share(uid, key, is_brand=True):
    tot = USER_TOTAL.get(uid, 0)
    if tot == 0: return 0.0
    cnt = USER_BRAND_CNT[(uid, key)] if is_brand else USER_CAT_CNT[(uid, key)]
    return float(cnt) / float(tot)
def _eq(a, b): return 1.0 if a == b else 0.0
def _recency_days(uid, ref_dt):
    last = USER_LAST_DATE.get(uid)
    if (last is None) or (ref_dt is None):
        return 0.0
    # Asigură compatibilitatea aware-aware
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    if ref_dt.tzinfo is None:
        ref_dt = ref_dt.replace(tzinfo=timezone.utc)
    return max(0.0, (ref_dt - last).days)

# ---------------- Trend & recency caches ----------------
CAND_TREND = {}             # (pid, window) -> z-score log1p(count)
CURRENT_POOL_LEN = defaultdict(int)
GLOBAL_POOL_MEAN = 50.0
DAYS_SINCE_ANCHOR_SEEN = defaultdict(float)

def _build_trend_counters(orders, windows=("10m","1h","24h")):
    from collections import defaultdict as _dd
    now = datetime.now(timezone.utc)  # AWARE
    cnt = {w: _dd(int) for w in windows}
    last_seen = _dd(lambda: None)

    for o in orders:
        odt = parse_dt(o.get("orderDate")) or now
        if odt.tzinfo is None:
            odt = odt.replace(tzinfo=timezone.utc)  # just in case

        for op in (o.get("orderProducts") or []):
            pid = op.get("productId")
            if pid is None:
                continue
            dt_sec = (now - odt).total_seconds()  # AWARE - AWARE
            if dt_sec <= 600:   cnt["10m"][pid] += 1
            if dt_sec <= 3600:  cnt["1h"][pid]  += 1
            if dt_sec <= 86400: cnt["24h"][pid] += 1
            last_seen[pid] = max(last_seen[pid], odt) if last_seen[pid] else odt

    for w in windows:
        vals = list(cnt[w].values())
        s = np.log1p(np.array(vals)) if vals else np.array([1.0])
        mu, sd = float(np.mean(s)), float(np.std(s) + 1e-9)
        for pid, v in cnt[w].items():
            CAND_TREND[(pid, w)] = (math.log1p(v) - mu) / sd

    for pid, dt_ in last_seen.items():
        # dt_ e deja AWARE -> scazi din now (tot AWARE)
        DAYS_SINCE_ANCHOR_SEEN[pid] = float((now - dt_).days)

_build_trend_counters(train_orders)

# ---------------- Financial features (fail-safe) ----------------
def _truthy(x):
    if isinstance(x, (int, float)): return float(x) != 0.0
    s = str(x).strip().lower()
    return s in {"1","true","yes","y","da","t","ok"}
def _gender_is_female(x):
    s = str(x).strip().lower()
    return 1.0 if s in {"f","female","feminin","femeie","woman","women"} else 0.0
def _num_from_range(v):
    if v is None: return 0.0
    if isinstance(v, (int, float)):
        try: return float(v)
        except: return 0.0
    s = str(v)
    nums = re.findall(r"[-+]?\d*\.?\d+", s)
    if len(nums) >= 2:
        try:
            a = float(nums[0]); b = float(nums[1])
            if a > b: a, b = b, a
            return 0.5*(a+b)
        except: pass
    try: return float(s)
    except: return 0.0
def _hash_cat(val):
    if val is None: return 0
    s = str(val).strip()
    if s == "" or s.lower() in {"nan","none","null","unknown"}: return 0
    return int(hash(s) % 1000)

def _load_financial_features(fin_path):
    if not os.path.exists(fin_path): return {}
    df = pd.read_csv(fin_path)
    if "ClientId" not in df.columns:
        for alt in ["clientId", "client_id", "userId", "user_id"]:
            if alt in df.columns: df = df.rename(columns={alt: "ClientId"}); break
    if "ClientId" not in df.columns:
        print("[FIN] Lipsă col ClientId — ignor features financiare.")
        return {}
    df["__key_str"] = df["ClientId"].astype(str)
    features_by_key = {}
    for _, row in df.iterrows():
        d = row.to_dict(); k = d.get("__key_str"); features_by_key[k] = d
        try:
            ki = int(d["ClientId"]); features_by_key[ki] = d
        except: pass
    print(f"[FIN] Încarcate {len(features_by_key)} profile financiare (fail-safe).")
    return features_by_key

FINANCIAL_FEATURES = _load_financial_features(FINANCIAL_PATH)
def _get_fin(uid):
    if uid in FINANCIAL_FEATURES: return FINANCIAL_FEATURES[uid]
    k = str(uid)
    if k in FINANCIAL_FEATURES: return FINANCIAL_FEATURES[k]
    try:
        ki = int(uid)
        if ki in FINANCIAL_FEATURES: return FINANCIAL_FEATURES[ki]
    except: pass
    return {}

# ---------------- Co-oc stats ----------------
RERANK_ONLY_LONGTAIL = True
def cooc_stats(a, b, LAMBDA_SHRINK=40.0):
    if a == b: return 0,0,0,0.0,0.0,0.0
    x,y = (a,b) if a<b else (b,a)
    supp = pair_count.get((x,y),0)
    ca, cb = item_count.get(a,0), item_count.get(b,0)
    if supp==0 or ca==0 or cb==0: return supp,ca,cb,0.0,0.0,0.0
    pa, pb, pab = ca/N_train_baskets, cb/N_train_baskets, supp/N_train_baskets
    lift = pab / (pa*pb + 1e-12)
    lift_c = min(lift, 5.0) / 5.0
    PMI = math.log((pab + 1e-12) / (pa*pb + 1e-12))
    NPMI = PMI / (-math.log(pab + 1e-12))
    npmi01 = max(0.0, min(1.0, 0.5*(NPMI+1.0)))
    jacc = supp / (ca + cb - supp + 1e-12)
    shrink = supp / (supp + LAMBDA_SHRINK)
    return supp,ca,cb, lift_c*shrink, npmi01*shrink, jacc*shrink

# ------- LGBM features + monotone constraints -------
_FIN_FEATS = [
    "fin_age","fin_is_female","fin_self_employed","fin_income","fin_budget",
    "fin_occupation","fin_county","fin_pref_prodtype","fin_fav_brand",
    "fin_fashion","fin_skin","fin_purchase_freq","fin_disc_sens",
    "fin_payment","fin_device","fin_social","fin_rewards"
]
_EXTRA_FEATS = [
    "cand_pop_w10m","cand_pop_w1h","cand_pop_w24h",
    "pool_len","pool_len_delta","days_since_anchor_seen",
    "user_cluster_id"
]
_LGBM_FEATS = [
    "cos_sim","lift_c","npmi01","jacc","supp",
    "pop_anchor","pop_cand","eq_brand","eq_cat",
    "user_brand_share","user_cat_share","user_recency_days",
] + _FIN_FEATS + _EXTRA_FEATS

def _featurize(uid, anchor_pid, cand_pid, cosv, ref_dt):
    a_meta = ITEM_META.get(anchor_pid, {"brand":"unknown","category":"unknown"})
    c_meta = ITEM_META.get(cand_pid,   {"brand":"unknown","category":"unknown"})
    a_b, a_c = a_meta["brand"], a_meta["category"]
    c_b, c_c = c_meta["brand"], c_meta["category"]
    supp, ca, cb, lift_c, npmi01, jacc = cooc_stats(anchor_pid, cand_pid, LAMBDA_SHRINK)
    pop_a, pop_c = _pop(anchor_pid), _pop(cand_pid)

    # trend & recency
    cand_w10m = CAND_TREND.get((cand_pid, "10m"), 0.0)
    cand_w1h  = CAND_TREND.get((cand_pid, "1h"), 0.0)
    cand_w24h = CAND_TREND.get((cand_pid, "24h"), 0.0)
    pool_len  = CURRENT_POOL_LEN.get(anchor_pid, 0)
    pool_delta = abs(pool_len - GLOBAL_POOL_MEAN)
    dsa = DAYS_SINCE_ANCHOR_SEEN.get(anchor_pid, 0.0)

    feats = [
        float(cosv),
        float(lift_c if supp >= 3 else 0.0),
        float(npmi01 if supp >= 3 else 0.0),
        float(jacc   if supp >= 3 else 0.0),
        float(supp),
        float(pop_a), float(pop_c),
        _eq(a_b, c_b), _eq(a_c, c_c),
        _share(uid, c_b, True), _share(uid, c_c, False),
        _recency_days(uid, ref_dt),
    ]

    # financial (fail-safe map)
    f = _get_fin(uid)
    feats.extend([
        float(_num_from_range(f.get("Age", 0))),
        _gender_is_female(f.get("Gender")),
        1.0 if _truthy(f.get("SelfEmployed")) else 0.0,
        float(_num_from_range(f.get("IncomeRange", 0))),
        float(_num_from_range(f.get("BudgetRange", 0))),
        _hash_cat(f.get("Occupation")),
        _hash_cat(f.get("County")),
        _hash_cat(f.get("PreferredProductTypes")),
        _hash_cat(f.get("FavoriteBrands")),
        _hash_cat(f.get("FashionStyle")),
        _hash_cat(f.get("SkinType")),
        _hash_cat(f.get("PurchaseFrequency")),
        _hash_cat(f.get("DiscountSensitivity")),
        _hash_cat(f.get("PreferredPaymentMethod")),
        _hash_cat(f.get("DeviceUsed")),
        _hash_cat(f.get("SocialMediaInfluence")),
        _hash_cat(f.get("PreferredRewards")),
    ])

    # extra recency/trend
    feats.extend([float(cand_w10m), float(cand_w1h), float(cand_w24h),
                  float(pool_len), float(pool_delta), float(dsa),
                  float(USER_CLUSTER.get(uid, -1))])
    return feats

# Monotone constraints (ordinea corespunde _LGBM_FEATS)
def _build_monotone():
    mono = []
    for f in _LGBM_FEATS:
        if f in {"cos_sim","lift_c","npmi01","jacc","supp",
                 "pop_cand","user_brand_share","user_cat_share",
                 "cand_pop_w10m","cand_pop_w1h","cand_pop_w24h"}:
            mono.append(1)   # crescător
        elif f in {"user_recency_days"}:
            mono.append(-1)  # descrescător
        else:
            mono.append(0)
    return mono

# ---------------- Rerank scoring (ensemble ready) ----------------
_LGBMS = []   # ensemble de boostere LightGBM
TAU=1.0; ALPHA=0.75; BETA=0.14; GAMMA=0.08; DELTA=0.02
POP_EPS=0.0; LAMBDA_SHRINK=40.0; CAND_POOL=200
MMR_ENABLE=True; MMR_LAMBDA=0.95; MMR_MIN_RATIO=2.0
POOL_BASE=35; POOL_EXTRA=65
POP_THRESHOLD = percentile_pop(60)

def conf_bonus(supp): return 1.0 / (1.0 + math.exp(-(supp - 3.0)))
def _is_longtail(pid): return item_count.get(int(pid), 0) < POP_THRESHOLD if RERANK_ONLY_LONGTAIL else True

def rescoring(anchor_pid, cand_pid, cosv, uid=None, ref_dt=None):
    # Ensemble LGBM dacă avem uid (features user-aware)
    if _LGBMS and uid is not None:
        feats = np.asarray([_featurize(uid, anchor_pid, cand_pid, float(cosv), ref_dt)], dtype=np.float32)
        preds = [m.predict(feats, num_iteration=getattr(m, "best_iteration", None)) for m in _LGBMS]
        return float(np.mean(preds))
    # Fallback clasic
    cosv = math.tanh(cosv / max(1e-9, TAU))
    supp, _, _, lift_c, npmi01, jacc = cooc_stats(anchor_pid, cand_pid, LAMBDA_SHRINK)
    if supp < 3: lift_c = 0.0; npmi01 = 0.0; jacc = 0.0
    base = ALPHA*cosv + BETA*lift_c + GAMMA*npmi01 + DELTA*jacc
    score = 0.92*base + 0.08*conf_bonus(supp)
    if MAX_ITEM_COUNT > 0:
        pop_norm = math.log1p(item_count.get(cand_pid, 0)) / math.log1p(MAX_ITEM_COUNT)
        score += POP_EPS * pop_norm
    return score

def _adaptive_pool_size(anchor_pid, base=POOL_BASE, extra=POOL_EXTRA):
    pop = item_count.get(anchor_pid, 0)
    pop_norm = math.log1p(pop) / math.log1p(MAX_ITEM_COUNT) if MAX_ITEM_COUNT>0 else 0.0
    return int(max(10, min(CAND_POOL, base + (1.0 - pop_norm) * extra)))

# ---------------- ALS/BM25 builder ----------------
def build_train_df(train_orders, half_life_days: float):
    now = datetime.now(timezone.utc)
    rows = []
    user_event_count = Counter()
    for o in train_orders:
        u = o.get("clientId");
        if u is None: continue
        user_event_count[u] += len([op for op in o.get("orderProducts", []) if op.get("productId") is not None])
    for o in train_orders:
        u = o.get("clientId")
        if u is None: continue
        odt = parse_dt(o.get("orderDate")) or now
        if odt.tzinfo is None: odt = odt.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (now - odt).days)
        adj = 0.75 if user_event_count[u] > 30 else (1.25 if user_event_count[u] < 5 else 1.0)
        hl = half_life_days * adj
        w_time = math.exp(-math.log(2.0) * (age_days / hl))
        seen = set()
        for op in o.get("orderProducts", []):
            pid = op.get("productId")
            if pid is None or pid in seen: continue
            seen.add(pid)
            qty = float(op.get("quantity", 1))
            w_qty = np.log1p(qty)
            rows.append((u, pid, float(w_qty * w_time)))
    if not rows: return None
    df = pd.DataFrame(rows, columns=["user","item","weight"])
    if MIN_USER_EVENTS > 1:
        uc = df.groupby("user")["item"].count()
        df = df[df["user"].isin(uc[uc>=MIN_USER_EVENTS].index)]
    if MIN_ITEM_EVENTS > 1:
        ic = df.groupby("item")["user"].count()
        df = df[df["item"].isin(ic[ic>=MIN_ITEM_EVENTS].index)]
    return df if len(df)>0 else None

def df_to_users_items(df):
    users = np.sort(df["user"].unique())
    items = np.sort(df["item"].unique())
    u2i = {u:i for i,u in enumerate(users)}
    p2j = {p:j for j,p in enumerate(items)}
    ui = df["user"].map(u2i).values
    ij = df["item"].map(p2j).values
    wt = df["weight"].astype(np.float32).values
    R_counts = sp.csr_matrix((wt, (ui, ij)), shape=(len(users), len(items)))
    return R_counts, users, items

_CACHE_RC  = {}  # hl -> (R_counts, users_arr, items_arr)
_CACHE_RIU = {}  # (id(R_counts_T),K1,B) -> bm25(items x users)
def make_R_counts(half_life_days: float):
    if half_life_days in _CACHE_RC: return _CACHE_RC[half_life_days]
    df = build_train_df(train_orders, half_life_days)
    assert df is not None and len(df)>0, "Nu există interacțiuni valide după filtrare."
    R_counts, users_arr, items_arr = df_to_users_items(df)
    print(f"[HL={half_life_days}] Users: {len(users_arr)} | Items: {len(items_arr)} | NNZ: {R_counts.nnz}")
    _CACHE_RC[half_life_days] = (R_counts, users_arr, items_arr)
    return _CACHE_RC[half_life_days]

def get_Riu_from_Rcounts(R_counts_T, K1, B):
    key = (id(R_counts_T), K1, B)
    if key in _CACHE_RIU: return _CACHE_RIU[key]
    Riu = bm25_weight(R_counts_T, K1=K1, B=B).tocsr().astype(np.float32)
    _CACHE_RIU[key] = Riu
    return Riu

def train_als_get_Vn_from_Riu(Riu, k, iters, reg, alpha, items_arr):
    Riu = Riu.tocsr(copy=True)
    Riu.data = 1.0 + float(alpha) * Riu.data   # Hu et al.
    als = implicit.als.AlternatingLeastSquares(
        factors=int(k), iterations=int(iters), regularization=float(reg),
        random_state=SEED, calculate_training_loss=False,
        use_gpu=USE_GPU, num_threads=NUM_THREADS
    )
    als.fit(Riu)  # (items x users)
    V = als.item_factors if als.item_factors.shape[0] == len(items_arr) else als.user_factors
    V = V.astype(np.float32)
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return Vn

# ---------------- Eval baskets ----------------
def _make_eval_baskets_for_items(items_index_local, sample_frac=1.0):
    item_index_local = {int(pid): int(i) for i, pid in enumerate(items_index_local)}
    eval_b = []
    for o in test_orders:
        uid = o.get("clientId"); odt = parse_dt(o.get("orderDate"))
        # odt deja AWARE; fallback defensiv:
        if odt and odt.tzinfo is None:
            odt = odt.replace(tzinfo=timezone.utc)
        s = {op.get("productId") for op in (o.get("orderProducts") or []) if op.get("productId") is not None}
        bb = {pid for pid in s if pid in item_index_local}
        if len(bb) >= MIN_BASKET_LEN:
            eval_b.append((uid, bb, odt))
    if 0.0 < float(sample_frac) < 1.0 and len(eval_b) > 0:
        m = max(1, int(len(eval_b) * float(sample_frac)))
        eval_b = random.sample(eval_b, m)
    return eval_b, item_index_local

def _mmr_select_on_pool(Vn, indices_pool, scores_pool, k, lambda_mmr=0.95):
    if not MMR_ENABLE or k <= 1 or len(indices_pool) == 0:
        order = np.argsort(scores_pool)[::-1]
        return list(indices_pool[order[:max(0, k)]].astype(int))
    V_pool = Vn[indices_pool]
    try:
        S_pool = V_pool @ V_pool.T
    except MemoryError:
        order = np.argsort(scores_pool)[::-1]
        return list(indices_pool[order[:max(0, k)]].astype(int))
    selected_pos = []
    cand_mask = np.ones(len(indices_pool), dtype=bool)
    for _ in range(min(k, len(indices_pool))):
        if not selected_pos:
            j = int(np.argmax(scores_pool)); selected_pos.append(j); cand_mask[j] = False; continue
        max_sim_sel = np.max(S_pool[:, selected_pos], axis=1)
        mmr_scores = lambda_mmr * scores_pool - (1.0 - lambda_mmr) * max_sim_sel
        mmr_scores[~cand_mask] = -1e9
        j = int(np.argmax(mmr_scores))
        if mmr_scores[j] <= -1e8: break
        selected_pos.append(j); cand_mask[j] = False
    return list(indices_pool[np.array(selected_pos, dtype=int)].astype(int))

def _pick_with_rerank(ii, Vn, items_index_local, sims_row, K, uid=None, ref_dt=None):
    n_items = sims_row.shape[0]
    kk = max(1, min(K, n_items-1))
    m = _adaptive_pool_size(int(items_index_local[ii]), base=POOL_BASE, extra=POOL_EXTRA)
    m = max(kk, min(m, n_items-1))
    pool = np.argpartition(sims_row, -m)[-m:]
    pool = pool[np.argsort(sims_row[pool])[::-1]]

    anchor_pid = int(items_index_local[ii])
    CURRENT_POOL_LEN[anchor_pid] = int(len(pool))  # pentru features

    if not _is_longtail(anchor_pid):
        return list(pool[:kk].astype(int))

    rescored_scores = np.asarray([
        rescoring(anchor_pid, int(items_index_local[int(j)]), float(sims_row[int(j)]), uid=uid, ref_dt=ref_dt)
        for j in pool
    ], dtype=np.float32)
    if MMR_ENABLE and (len(pool) >= int(MMR_MIN_RATIO*kk)):
        return _mmr_select_on_pool(Vn, pool, rescored_scores, kk, lambda_mmr=MMR_LAMBDA)
    else:
        order = np.argsort(rescored_scores)[::-1]
        return list(pool[order[:kk]].astype(int))

def _eval_for_K(Vn, K, items_index_local, eval_baskets_local, item_index_local, use_rerank=True):
    total=hits=0; precisions=[]; recalls=[]; ap_list=[]; rr_list=[]; rec_glob=set()
    n_items = Vn.shape[0]
    if n_items <= 1:
        return {"cases":0, "HitRate@K":0, "Precision@K":0, "Recall@K":0, "MAP@K":0, "MRR@K":0, "Coverage@K":0}
    kk_global = max(1, min(K, n_items-1))
    for uid, basket, odt in eval_baskets_local:
        for anchor in basket:
            targets = set(basket) - {anchor}
            if not targets: continue
            total += 1
            ii = item_index_local.get(int(anchor))
            if ii is None: continue
            sims_row = Vn @ Vn[ii]; sims_row[ii] = -1.0
            if use_rerank:
                chosen_js = _pick_with_rerank(ii, Vn, items_index_local, sims_row, kk_global, uid=uid, ref_dt=odt)
            else:
                top = np.argpartition(sims_row, -kk_global)[-kk_global:]
                chosen_js = top[np.argsort(sims_row[top])[::-1]]
            rec_ids = [int(items_index_local[int(j)]) for j in chosen_js]
            rec_glob.update(rec_ids)
            inter = targets.intersection(rec_ids)
            if inter: hits += 1
            precisions.append(len(inter)/kk_global)
            recalls.append(len(inter)/len(targets))
            cum_rel=sum_prec=0.0
            for rank,pid in enumerate(rec_ids, start=1):
                if pid in targets: cum_rel+=1; sum_prec+=cum_rel/rank
            denom = min(len(targets), kk_global)
            ap_list.append(sum_prec/denom if denom>0 else 0.0)
            rr=0.0
            for rank,pid in enumerate(rec_ids, start=1):
                if pid in targets: rr=1.0/rank; break
            rr_list.append(rr)
    cov = len(rec_glob)/len(items_index_local) if len(items_index_local) else 0.0
    return {
        "cases": int(total),
        "HitRate@K": float(hits/total) if total else 0.0,
        "Precision@K": float(np.mean(precisions)) if precisions else 0.0,
        "Recall@K": float(np.mean(recalls)) if recalls else 0.0,
        "MAP@K": float(np.mean(ap_list)) if ap_list else 0.0,
        "MRR@K": float(np.mean(rr_list)) if rr_list else 0.0,
        "Coverage@K": float(cov),
    }

def evaluate_model(Vn, items_index_local, sample_frac=1.0):
    eval_baskets_local, item_index_local = _make_eval_baskets_for_items(items_index_local, sample_frac=sample_frac)
    out = {}
    for K in EVAL_TOPKS:
        out[f"BASE K={K}"]   = _eval_for_K(Vn, K, items_index_local, eval_baskets_local, item_index_local, use_rerank=False)
        out[f"RERANK K={K}"] = _eval_for_K(Vn, K, items_index_local, eval_baskets_local, item_index_local, use_rerank=True)
    return out

def pick_key(metrics_dict, primary_k=PRIMARY_K):
    m = metrics_dict.get(f"RERANK K={primary_k}", {})
    return (m.get("MAP@K",0.0), m.get("Precision@K",0.0), m.get("Recall@K",0.0))

# ---------------- (Optional) user clustering ----------------
USER_CLUSTER = defaultdict(lambda: -1)
try:
    from sklearn.cluster import MiniBatchKMeans
    u2b = defaultdict(list); u2c = defaultdict(list)
    for (uid, b), cnt in USER_BRAND_CNT.items(): u2b[uid].append(cnt)
    for (uid, c), cnt in USER_CAT_CNT.items():   u2c[uid].append(cnt)
    U, Xu = [], []
    for uid in USER_TOTAL.keys():
        bs = float(np.mean(u2b[uid])) if u2b[uid] else 0.0
        cs = float(np.mean(u2c[uid])) if u2c[uid] else 0.0
        U.append(uid); Xu.append([bs, cs])
    if Xu:
        km = MiniBatchKMeans(n_clusters=12, n_init=10, random_state=SEED, batch_size=2048)
        lab = km.fit_predict(np.asarray(Xu, dtype=np.float32))
        USER_CLUSTER = {u:int(l) for u,l in zip(U, lab)}
        print(f"[CLUST] user clusters: {len(set(lab))}")
except Exception as e:
    print(f"[CLUST] skip: {e}")

# ---------------- LGBM: build train data & train ----------------
def build_rerank_train(Vn, items_index_local, mmr_pool=200, neg_per_pos=3, max_pos_per_query=6):
    item_index_local = {int(pid): int(i) for i, pid in enumerate(items_index_local)}
    rows = []
    train_baskets_uid = []
    for o in train_orders:
        uid = o.get("clientId"); odt = parse_dt(o.get("orderDate"))
        if uid is None or odt is None: continue
        # odt este AWARE prin parse_dt; fallback defensiv:
        if odt.tzinfo is None:
            odt = odt.replace(tzinfo=timezone.utc)
        s = {op.get("productId") for op in (o.get("orderProducts") or []) if op.get("productId") is not None}
        s = {pid for pid in s if pid in item_index_local}
        if len(s) >= MIN_BASKET_LEN:
            train_baskets_uid.append((uid, s, odt))

    def _hardness(pid_anchor, pid_cand, raw_sim):
        s, _, _, lift_c, npmi01, jacc = cooc_stats(pid_anchor, pid_cand, LAMBDA_SHRINK)
        cooc = max(lift_c, npmi01, jacc)
        return 0.7*float(raw_sim) + 0.3*float(cooc)

    for uid, basket, odt in train_baskets_uid:
        for anchor in basket:
            ii = item_index_local.get(int(anchor))
            if ii is None: continue
            targets = list(basket - {anchor})
            if not targets: continue
            sims = Vn @ Vn[ii]; sims[ii] = -1.0
            m = max(len(targets)*neg_per_pos + 10, mmr_pool//4)
            m = min(max(m, 20), Vn.shape[0]-1)
            pool = np.argpartition(sims, -m)[-m:]
            pool = pool[np.argsort(sims[pool])[::-1]]
            targets = targets[:max_pos_per_query]
            pool_cands = [int(items_index_local[int(j)]) for j in pool if int(items_index_local[int(j)]) not in basket]
            pool_cands_scored = sorted(pool_cands, key=lambda p: _hardness(anchor, p, sims[item_index_local[p]]), reverse=True)
            neg_take = min(len(pool_cands_scored), len(targets)*neg_per_pos)
            negs = pool_cands_scored[:neg_take]
            qid = f"{uid}::{anchor}"
            for p in targets:
                rows.append((qid, 1, _featurize(uid, anchor, p, sims[item_index_local[p]], odt)))
            for n in negs:
                rows.append((qid, 0, _featurize(uid, anchor, n, sims[item_index_local[n]], odt)))

    if not rows: return None, None, None, None

    df = pd.DataFrame({
        "qid": [r[0] for r in rows],
        "y":   [r[1] for r in rows],
        **{f: [r[2][i] for r in rows] for i, f in enumerate(_LGBM_FEATS)}
    }).sort_values("qid").reset_index(drop=True)

    # group weights by recency (1/(1+days) pe query) — FIX timezone-aware
    qid_weights = {}
    now_utc = datetime.now(timezone.utc)
    for uid, basket, odt in train_baskets_uid:
        for anchor in basket:
            qid = f"{uid}::{anchor}"
            odt_aware = odt if (odt and odt.tzinfo is not None) else now_utc
            age_days = max(0.0, (now_utc - odt_aware).days)
            w = 1.0 / (1.0 + age_days)
            qid_weights[qid] = max(qid_weights.get(qid, 0.0), w)

    X = df[_LGBM_FEATS].values.astype(np.float32)
    y = df["y"].values.astype(np.int32)
    group = df.groupby("qid").size().values.astype(np.int32)
    weights = df["qid"].map(qid_weights).fillna(1.0).values.astype(np.float32)
    return X, y, group, weights

def _lgbm_base_params():
    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
        "learning_rate": 0.03,
        "num_leaves": 127,
        "max_depth": -1,
        "min_data_in_leaf": 80,
        "min_sum_hessian_in_leaf": 1e-3,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 1e-3,
        "lambda_l2": 1e-2,
        "min_gain_to_split": 0.0,
        "max_bin": 255,
        "verbose": -1,
        "deterministic": True,
        "force_row_wise": True,
        "random_state": 42,
        "monotone_constraints": _build_monotone(),
    }

def _optuna_tune(X_train, y_train, g_train, X_val, y_val, g_val, base, trials=20):
    import optuna
    def _obj(trial):
        params = dict(base)
        params.update({
            "num_leaves": trial.suggest_int("num_leaves", 63, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 0.95),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 0.95),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 1e-1, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 1e-1, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.05),
        })
        dtr = lgb.Dataset(X_train, label=y_train, group=g_train, free_raw_data=True)
        dvl = lgb.Dataset(X_val,   label=y_val,   group=g_val,   free_raw_data=True)
        bst = lgb.train(params, dtr, valid_sets=[dvl], num_boost_round=800,
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        return float(bst.best_score["valid_0"]["ndcg@5"])
    study = optuna.create_study(direction="maximize")
    study.optimize(_obj, n_trials=trials)
    print("[HPO] Best:", study.best_value, study.best_params)
    out = dict(base); out.update(study.best_params)
    return out

def train_lgbm_reranker(Vn, items_index_local, seed=42):
    data = build_rerank_train(Vn, items_index_local, mmr_pool=200, neg_per_pos=3, max_pos_per_query=6)
    if data[0] is None:
        print("[LGBM] Nu am putut construi datele.")
        return None
    X, y, group, weights = data
    # split pe queries (80/20) păstrând grupurile contigue
    cum = np.cumsum(group)
    target = int(0.8 * cum[-1])
    n_train_groups = int(np.searchsorted(cum, target, side="right"))
    train_end = int(cum[n_train_groups-1]) if n_train_groups > 0 else 0
    g_train, g_val = group[:n_train_groups], group[n_train_groups:]
    X_train, X_val = X[:train_end], X[train_end:]
    y_train, y_val = y[:train_end], y[train_end:]
    w_train, w_val = weights[:train_end], weights[train_end:]

    base = _lgbm_base_params(); base["random_state"] = seed
    if USE_OPTUNA:
        tuned = _optuna_tune(X_train, y_train, g_train, X_val, y_val, g_val, base=base, trials=20)
    else:
        tuned = base

    dtrain = lgb.Dataset(X_train, label=y_train, group=g_train, weight=w_train, free_raw_data=True)
    dvalid = lgb.Dataset(X_val,   label=y_val,   group=g_val,   weight=w_val,   free_raw_data=True)

    booster = lgb.train(
        tuned, dtrain, valid_sets=[dvalid],
        num_boost_round=1200,
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(100)]
    )
    print(f"[LGBM] seed={seed}  best_iter={booster.best_iteration}  ndcg5={booster.best_score['valid_0']['ndcg@5']:.6f}")
    return booster

# ---------------- Precompute vecini pt best ----------------
def precompute_neighbors(Vn, items_index_local):
    n_items = Vn.shape[0]
    precomp = {}
    if n_items <= 1: return precomp
    for ii in range(n_items):
        anchor_pid = int(items_index_local[ii])
        sims_row = Vn @ Vn[ii]; sims_row[ii] = -1.0
        kk = TOP_NEIGHBORS
        m = _adaptive_pool_size(anchor_pid, base=POOL_BASE, extra=POOL_EXTRA)
        m = max(kk, min(m, n_items-1))
        pool = np.argpartition(sims_row, -m)[-m:]
        pool = pool[np.argsort(sims_row[pool])[::-1]]
        out = []
        for j in pool[:kk]:
            pid = int(items_index_local[int(j)])
            if _LGBMS:
                sc = rescoring(anchor_pid, pid, float(sims_row[int(j)]), uid=None, ref_dt=None)
            else:
                sc = float(sims_row[int(j)])
            out.append((pid, float(np.float16(sc))))
        precomp[int(anchor_pid)] = out
    return precomp

# ---------------- Persist helpers ----------------
def save_results_row(params, metrics, key_tuple, csv_path=RESULTS_CSV, jsonl_path=RESULTS_JSONL):
    row = dict(params)
    mR = metrics.get(f"RERANK K={PRIMARY_K}", {})
    mB = metrics.get(f"BASE K={PRIMARY_K}", {})
    for prefix, md in [("base", mB), ("rerank", mR)]:
        for k,v in md.items(): row[f"{prefix}_{k}"] = v
    row["score_MAP"]    = float(key_tuple[0])
    row["score_Prec"]   = float(key_tuple[1])
    row["score_Recall"] = float(key_tuple[2])
    row["timestamp"]    = int(time.time())
    df = pd.DataFrame([row])
    if not os.path.exists(csv_path): df.to_csv(csv_path, index=False)
    else: df.to_csv(csv_path, mode="a", header=False, index=False)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(_json.dumps(row)+"\n")

def save_best_artifact(Vn, items_index_local, params, precomp, item_count_local, best_pkl=BEST_PKL):
    TOP_POPULAR = [int(pid) for pid,_ in sorted(item_count_local.items(), key=lambda x: x[1], reverse=True)[:TOP_NEIGHBORS]]
    artifact = {
        "item_index": {int(pid): int(i) for i, pid in enumerate(items_index_local)},
        "index_item": {int(i): int(pid) for i, pid in enumerate(items_index_local)},
        "items_index": items_index_local.tolist(),
        "norm_item_factors": Vn.astype(np.float16),
        "params": params,
        "precomputed_neighbors_reranked": {"top": int(TOP_NEIGHBORS), "neighbors": precomp},
        "top_popular_fallback": TOP_POPULAR
    }
    with open(best_pkl, "wb") as f: pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[OK] Best model salvat: {best_pkl}")

# ---------------- Resume helpers ----------------
def params_fingerprint(hl, bm25, als_idx, rr_idx):
    cfg = ALS_GRID[als_idx]
    s = f"hl={hl}|k1={bm25[0]}|b={bm25[1]}|k={cfg['K']}|it={cfg['IT']}|reg={cfg['REG']}|a={cfg['ALPHA']}|rr={rr_idx}"
    return hashlib.md5(s.encode()).hexdigest()

def read_done_fingerprints(csv_path=RESULTS_CSV):
    if not os.path.exists(csv_path): return set()
    try:
        df = pd.read_csv(csv_path, usecols=["half_life","bm25_K1","bm25_B","K","IT","REG","ALPHA","rerank_profile"])
        done = set()
        for _,r in df.iterrows():
            s = f"hl={r['half_life']}|k1={r['bm25_K1']}|b={r['bm25_B']}|k={r['K']}|it={r['IT']}|reg={r['REG']}|a={r['ALPHA']}|rr={int(r['rerank_profile'])}"
            done.add(hashlib.md5(s.encode()).hexdigest())
        return done
    except Exception:
        return set()

# ---------------- Main sweep (coarse → refine) ----------------
def main():
    global RERANK_ONLY_LONGTAIL, POP_THRESHOLD
    global ALPHA,BETA,GAMMA,DELTA,TAU,POP_EPS,LAMBDA_SHRINK,CAND_POOL
    global MMR_ENABLE,MMR_LAMBDA,MMR_MIN_RATIO,POOL_BASE,POOL_EXTRA
    global _LGBMS

    combos_full = list(product(HALF_LIFE_GRID, BM25_GRID, range(len(ALS_GRID)), range(len(RERANK_PROFILES))))
    print(f"[INFO] Combinații totale: {len(combos_full)}")

    if FAST_SWEEP:
        coarse_HL = [45.0, 90.0, 180.0, 240.0]
        coarse_BM25 = [(1.2,0.75), (1.4,0.75), (1.6,0.75), (1.4,0.85)]
        coarse_ALS_idx = [0, 2, 3, 4, 5]
        coarse_RR_idx = [0, 1, 2]
        combos = list(product(coarse_HL, coarse_BM25, coarse_ALS_idx, coarse_RR_idx))
        if len(combos) > MAX_TRIALS:
            random.shuffle(combos); combos = combos[:MAX_TRIALS]
        print(f"[FAST] Stage-1: testez {len(combos)} combinații (din {len(combos_full)})")
        sample_frac = EVAL_SAMPLE_FRAC
    else:
        combos = combos_full; sample_frac = 1.0

    done = read_done_fingerprints(RESULTS_CSV)
    trials_run = 0
    best = None
    stage1_scores = []
    t0 = time.time()

    for t_idx, (hl, (K1,B), als_idx, rr_idx) in enumerate(combos, start=1):
        fp = params_fingerprint(hl, (K1,B), als_idx, rr_idx)
        if fp in done: continue
        cfg = ALS_GRID[als_idx]
        iters_used = min(cfg["IT"], ALS_IT_CAP) if FAST_SWEEP and ALS_IT_CAP else cfg["IT"]
        print(f"\n===== Stage-1 Trial {t_idx}/{len(combos)} | HL={hl} | BM25(K1={K1},B={B}) | ALS={{K:{cfg['K']},IT:{iters_used},REG:{cfg['REG']},A:{cfg['ALPHA']}}} | RR#{rr_idx} =====")

        prof = RERANK_PROFILES[rr_idx]
        RERANK_ONLY_LONGTAIL = bool(prof["RERANK_ONLY_LONGTAIL"])
        POP_THRESHOLD = percentile_pop(int(prof["POP_PCTL"]))
        ALPHA=prof["ALPHA"]; BETA=prof["BETA"]; GAMMA=prof["GAMMA"]; DELTA=prof["DELTA"]
        TAU=prof["TAU"]; POP_EPS=prof["POP_EPS"]; LAMBDA_SHRINK=prof["LAMBDA_SHRINK"]; CAND_POOL=prof["CAND_POOL"]
        MMR_ENABLE=bool(prof["MMR_ENABLE"]); MMR_LAMBDA=prof["MMR_LAMBDA"]; MMR_MIN_RATIO=prof["MMR_MIN_RATIO"]
        POOL_BASE=prof["POOL_BASE"]; POOL_EXTRA=prof["POOL_EXTRA"]

        try:
            R_counts, users_arr, items_arr = make_R_counts(hl)
            R_counts_T = R_counts.T
            Riu_base = get_Riu_from_Rcounts(R_counts_T, K1, B)
            Riu = Riu_base.copy()
            base_conf = Riu.data.copy()
            Riu.data = 1.0 + float(cfg["ALPHA"]) * base_conf

            Vn_try = train_als_get_Vn_from_Riu(Riu=Riu, k=cfg["K"], iters=iters_used, reg=cfg["REG"], alpha=cfg["ALPHA"], items_arr=items_arr)
            items_index_local = np.asarray(items_arr, dtype=np.int64)

            metrics = evaluate_model(Vn_try, items_index_local, sample_frac=sample_frac)
            key = pick_key(metrics, primary_k=PRIMARY_K)

            params = {
                "seed": SEED, "use_only_completed": USE_ONLY_COMPLETED,
                "split": "leave-last-per-user", "test_split": TEST_SPLIT,
                "min_user_events": MIN_USER_EVENTS, "min_item_events": MIN_ITEM_EVENTS,
                "half_life": hl, "bm25_K1": K1, "bm25_B": B, **cfg,
                "rerank_profile": rr_idx,
                "rerank_alpha": ALPHA, "rerank_beta": BETA, "rerank_gamma": GAMMA, "rerank_delta": DELTA,
                "tau": TAU, "pop_eps": POP_EPS, "lambda_shrink": LAMBDA_SHRINK, "cand_pool": CAND_POOL,
                "mmr_enable": MMR_ENABLE, "mmr_lambda": MMR_LAMBDA, "mmr_min_ratio": MMR_MIN_RATIO,
                "pool_base": POOL_BASE, "pool_extra": POOL_EXTRA,
                "rerank_only_longtail": RERANK_ONLY_LONGTAIL, "pop_pctl": prof["POP_PCTL"],
                "eval_topks": list(EVAL_TOPKS),
                "stage": "coarse", "iters_used": iters_used, "sample_frac": sample_frac
            }
            save_results_row(params, metrics, key)
            print(f"[Stage-1 score @K={PRIMARY_K}] MAP={key[0]:.6f} | Prec={key[1]:.6f} | Rec={key[2]:.6f}")

            stage1_scores.append((key, (hl,(K1,B),als_idx,rr_idx)))

            if (best is None) or (key > best["key"]):
                best = {"key": key, "params": params, "Vn": Vn_try, "items_index": items_index_local}
                print(f"[BEST↑] Nou best (coarse) @K={PRIMARY_K}: MAP={key[0]:.6f} Prec={key[1]:.6f} Rec={key[2]:.6f}")

            trials_run += 1
            if (best is not None) and (trials_run % SAVE_INTERMEDIATE_BEST_EVERY == 0):
                print("[CKPT] Salvez checkpoint best intermediar…")
                precomp_ck = precompute_neighbors(best["Vn"], best["items_index"])
                save_best_artifact(best["Vn"], best["items_index"], best["params"], precomp_ck, item_count, best_pkl=TEMP_PKL)

        except AssertionError as e:
            print(f"[WARN] Trial sărit: {e}")
        except Exception as e:
            print(f"[ERROR] Trial failure: {type(e).__name__}: {e}")

    assert best is not None, "Stage-1 nu a produs niciun setup valid."

    # 2) Stage-2: refine TOPN_REFINE (full eval, full iters)
    stage1_scores.sort(key=lambda x: x[0], reverse=True)
    refine_list = stage1_scores[:TOPN_REFINE]
    print(f"\n[REFINE] Re-evaluez complet TOP {len(refine_list)} configurații…")

    for rank_idx, (key_coarse, combo) in enumerate(refine_list, start=1):
        hl, (K1,B), als_idx, rr_idx = combo
        cfg = ALS_GRID[als_idx]
        print(f"\n----- Refine {rank_idx}/{len(refine_list)} | HL={hl} | BM25(K1={K1},B={B}) | ALS={cfg} | RR#{rr_idx} -----")

        prof = RERANK_PROFILES[rr_idx]
        RERANK_ONLY_LONGTAIL = bool(prof["RERANK_ONLY_LONGTAIL"])
        POP_THRESHOLD = percentile_pop(int(prof["POP_PCTL"]))
        ALPHA=prof["ALPHA"]; BETA=prof["BETA"]; GAMMA=prof["GAMMA"]; DELTA=prof["DELTA"]
        TAU=prof["TAU"]; POP_EPS=prof["POP_EPS"]; LAMBDA_SHRINK=prof["LAMBDA_SHRINK"]; CAND_POOL=prof["CAND_POOL"]
        MMR_ENABLE=bool(prof["MMR_ENABLE"]); MMR_LAMBDA=prof["MMR_LAMBDA"]; MMR_MIN_RATIO=prof["MMR_MIN_RATIO"]
        POOL_BASE=prof["POOL_BASE"]; POOL_EXTRA=prof["POOL_EXTRA"]

        try:
            R_counts, users_arr, items_arr = make_R_counts(hl)
            R_counts_T = R_counts.T
            Riu_base = get_Riu_from_Rcounts(R_counts_T, K1, B)
            Riu = Riu_base.copy()
            base_conf = Riu.data.copy()
            Riu.data = 1.0 + float(cfg["ALPHA"]) * base_conf

            Vn_try = train_als_get_Vn_from_Riu(Riu=Riu, k=cfg["K"], iters=cfg["IT"], reg=cfg["REG"], alpha=cfg["ALPHA"], items_arr=items_arr)
            items_index_local = np.asarray(items_arr, dtype=np.int64)

            metrics = evaluate_model(Vn_try, items_index_local, sample_frac=1.0)
            key_full = pick_key(metrics, primary_k=PRIMARY_K)

            params = {
                **best["params"],
                "half_life": hl, "bm25_K1": K1, "bm25_B": B, **cfg,
                "rerank_profile": rr_idx,
                "stage": "refine", "iters_used": cfg["IT"], "sample_frac": 1.0
            }
            save_results_row(params, metrics, key_full)
            print(f"[Refine score @K={PRIMARY_K}] MAP={key_full[0]:.6f} | Prec={key_full[1]:.6f} | Rec={key_full[2]:.6f}")

            if key_full > best["key"]:
                best = {"key": key_full, "params": params, "Vn": Vn_try, "items_index": items_index_local}
                print(f"[BEST↑] Nou best (refine) @K={PRIMARY_K}: MAP={key_full[0]:.6f} Prec={key_full[1]:.6f} Rec={key_full[2]:.6f}")

        except Exception as e:
            print(f"[ERROR-REFINE] {type(e).__name__}: {e}")

    # 2.5) LGBM ensemble pe seed-uri
    _LGBMS = []
    if TRAIN_LGBM_RERANKER:
        print("\n[LGBM] Antrenez ensemble (multi-seed) pe configurația câștigătoare…")
        for s in ENSEMBLE_SEEDS:
            b = train_lgbm_reranker(best["Vn"], best["items_index"], seed=s)
            if b is not None: _LGBMS.append(b)
        if _LGBMS:
            metrics_lgbm = evaluate_model(best["Vn"], best["items_index"], sample_frac=1.0)
            key_lgbm = pick_key(metrics_lgbm, primary_k=PRIMARY_K)
            print(f"[LGBM eval @K={PRIMARY_K}] MAP={key_lgbm[0]:.6f} | Prec={key_lgbm[1]:.6f} | Rec={key_lgbm[2]:.6f}")

    # 3) Precompute vecini + salvează artefact
    print("\n[PRECOMPUTE] Calculez vecinii rerankați pentru configurația câștigătoare…")
    precomp = precompute_neighbors(best["Vn"], best["items_index"])
    save_best_artifact(best["Vn"], best["items_index"], best["params"], precomp, item_count, best_pkl=BEST_PKL)
    # Bought-Together summary (din TRAIN)
    bt_rules = bought_together_rules(min_support=10, min_conf=0.08, min_lift=1.05, top=20)
    print_bt_summary(bt_rules)


    dt = time.time() - t0
    sc = best["key"]
    print(f"\n[FINISH] Best @K={PRIMARY_K}: MAP={sc[0]:.6f} Prec={sc[1]:.6f} Rec={sc[2]:.6f}")
    print(f"[FINISH] Timp total (sec): {int(dt)}")
    print(f"[FINISH] Rezultate: {RESULTS_CSV} + {RESULTS_JSONL} | Artefact .pkl: {BEST_PKL}")

# ---------------- Inference helpers ----------------
def load_model(pkl_path=BEST_PKL):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def recommend_for_product(model, product_id, k=5):
    pid = int(product_id)
    pre = model.get("precomputed_neighbors_reranked", {}).get("neighbors", {})
    if pid in pre and pre[pid]:
        return pre[pid][:k]
    Vn = model["norm_item_factors"].astype(np.float32)
    items_index = np.asarray(model.get("items_index", []), dtype=np.int64)
    item_index = model["item_index"]
    if pid not in item_index:
        tops = model.get("top_popular_fallback", [])[:k]
        return [(int(p), 0.0) for p in tops]
    ii = item_index[pid]
    sims = Vn @ Vn[ii]; sims[ii] = -1.0
    kk = max(1, min(k, Vn.shape[0]-1))
    top = np.argpartition(sims, -kk)[-kk:]; top = top[np.argsort(sims[top])[::-1]]
    return [(int(items_index[int(j)]), float(sims[int(j)])) for j in top]

if __name__ == "__main__":
    main()
