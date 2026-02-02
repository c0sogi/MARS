import os
import numpy as np

# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_5"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# File Paths
TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")
ARTICLES_PATH = os.path.join(INPUT_DIR, "articles.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache File Paths (Parquet/NPY)
CACHE_TRANSACTIONS_PROCESSED = os.path.join(
    WORKING_DIR, "transactions_processed_fixed.parquet"
)
CACHE_MATRICES_HYBRID = os.path.join(WORKING_DIR, "hybrid_similarity_matrix_fixed.npz")
CACHE_USER_HISTORY = os.path.join(WORKING_DIR, "user_history_vectors_fixed.npz")
CACHE_GLOBAL_TRENDS = os.path.join(WORKING_DIR, "global_trends_fixed.parquet")
CACHE_ITEM_MAP = os.path.join(WORKING_DIR, "item_id_map_fixed.parquet")
CACHE_USER_MAP = os.path.join(WORKING_DIR, "user_id_map_fixed.parquet")

# =============================================================================
# GLOBAL PARAMETERS
# =============================================================================
SEED = 42
NUM_THREADS = 12  # Matches available vCPUs
PRECISION = np.float32  # Enforce float32 as per requirements

# =============================================================================
# DATA HYPERPARAMETERS
# =============================================================================
# Temporal Windowing: Restrict training data to the last N weeks
TRAIN_WEEKS = 5

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# 1. Stratum 1: Habitual Repurchase
# Logic: Score = 1 / days_elapsed
# No specific hyperparameters needed for the formula itself, but we define the offset.

# 2. Stratum 2: Hybrid Graph Retrieval
# S_hybrid = S_behavior + HYBRID_LAMBDA * S_variant
HYBRID_LAMBDA = 0.1

# 3. Stratum 3: Global Trend
# Time decay factor for calculating general popularity
TREND_DECAY_ALPHA = 0.95

# =============================================================================
# STRATIFICATION & SCORING (The Cascade)
# =============================================================================
# The model maps signals to disjoint score ranges to enforce hierarchy.
# Range 1: History [1000, inf)
# Range 2: CF [10, 900]
# Range 3: Trends [0, 9]

SCORE_OFFSET_HISTORY = 1000.0
SCORE_OFFSET_CF = 10.0
SCORE_OFFSET_TREND = 0.0

# Scaling factor to ensure CF scores (typically 0-1) fill the [10, 900] bucket
# 1.0 * 800 + 10 = 810, which fits safely below 1000.
CF_SCALING_FACTOR = 800.0

# Scaling factor for trends to fit in [0, 9]
TREND_SCALING_FACTOR = (
    1.0  # Assumes trend scores are normalized or small enough, or capped
)

# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================
TOP_K = 12
BATCH_SIZE = 5000  # Batch size for vectorized matrix multiplication to manage RAM
