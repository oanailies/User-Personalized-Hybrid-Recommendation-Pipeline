User-Personalized Hybrid Recommendation Pipeline

The User-Personalized Hybrid Recommendation Pipeline is a recommendation system designed for authenticated users, providing personalized product suggestions based on a combination of user behavior, item characteristics, and transaction history. Unlike simple item-to-item recommenders, this pipeline adds user-awareness, advanced personalization, and intelligent reranking to deliver more relevant and diverse recommendations.

What It Does

This pipeline generates “Products Suggested for You” lists tailored to each user by:

Learning hidden user and item preferences via ALS (Alternating Least Squares) latent factors.

Measuring how often items are bought together using co-occurrence metrics (lift, Jaccard, NPMI).

Incorporating recent user interactions more heavily through BM25 weighting and recency decay.

Adding interpretability via rule-based Bought Together signals, which identify complementary products.

Using a LightGBM LambdaRank reranker to combine all signals, including user preferences, item metadata, trend signals, and financial data, for the final recommendation ranking.

Handling cold-start scenarios for new users or items using fallback strategies based on category/brand popularity or user clustering.

Data Inputs

Transactional Data: Only completed orders are considered. Data is split using leave-last-per-user for realistic evaluation.

Product Data: Metadata such as brand, category, and other attributes.

User Data: Extracted from surveys and purchase history, including brand/category share, last purchase date, financial metadata (age, income, budget), and behavioral features.

Key Features and Components

ALS Latent Vectors: Captures hidden preferences for items and users from implicit interactions.

Co-occurrence Statistics: Quantifies associations between items in the same basket.

Bought Together Rules: Identifies frequently co-purchased item pairs for interpretability.

Trend Signals: Detects popular products over short-term windows (10 min, 1 hour, 24 hours).

Recency Weighting: Gives higher importance to recent interactions.

User Clustering: Groups users using MiniBatchKMeans, providing cluster-based features to improve ranking.

MMR Reranking: Ensures recommendation diversity and promotes long-tail items.

LightGBM Ranker

The final recommendation ranking is produced by a LightGBM LambdaRank model:

Optimized for nDCG@K to ensure top recommendations are relevant.

Applies monotone constraints to guarantee features with known effects influence predictions consistently.

Uses hard negatives—items similar to purchased ones but not bought—to increase discriminative power.

Inputs include ALS similarity, co-occurrence, trends, recency, item metadata, user features, and financial data.

Cold-Start Handling

New users or items with limited interactions are handled using fallback recommendations from popular items in the same category or brand.

User clusters provide additional signals to improve recommendations for new or sparse users.

Output

Personalized “Products Suggested for You” lists.

Supports real-time recommendation and batch processing.

Model artifacts include:

ALS latent vectors

User clusters

Precomputed item neighbors

LightGBM Ranker model

Example Bought Together Rules

Lipikar AP+ Oil Cleanser → Ginseng Cleansing Oil: Confidence 93.18%, Lift 131.65 → Strong complementary purchase.

Clinique For Men Hydrating Concentrate → Watermelon Glow SPF 30 Sunscreen: Confidence >92%, Support 256 → Reflects typical skincare routines.

False Eyelash Adhesive → Hoodie’s Lash #22 False Eyelashes: Confidence 91%, Support 243 → Frequently purchased together.

Exfoliating Scrub / Apricot Peeling Gel → AHA 30% + BHA 2% Peeling Solution: Lift >150 → Strong category association.

Bouncy & Firm Sleeping Mask ↔ Water Sleeping Mask: Confidence ~90%, Support 188 → Users prefer complementary night products.

These rules, derived from transaction data, demonstrate robustness and interpretability, and appear consistently across multiple recommendation pipelines.
