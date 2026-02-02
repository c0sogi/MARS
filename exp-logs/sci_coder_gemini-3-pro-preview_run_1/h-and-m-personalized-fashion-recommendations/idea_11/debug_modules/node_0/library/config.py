import os
import numpy as np

# ==========================================
# Global Configuration
# ==========================================

# Reproducibility
RANDOM_SEED = 42

# ==========================================
# File Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_11")
SUBMISSION_DIR = "./submission"

# Ensure necessary writeable directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Input Files
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

ARTICLES_PATH = os.path.join(INPUT_DIR, "articles.csv")
CUSTOMERS_PATH = os.path.join(INPUT_DIR, "customers.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output Files
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Data Processing Hyperparameters
# ==========================================

# Temporal Windowing
# We use a 20-week window to capture latent connections between rare items
# while relying on exponential decay to handle concept drift.
TRAIN_WEEKS = 20

# Validation
VAL_WEEKS = 1

# Time Decay Logic
# Formula: weight = exp(-lambda * days_elapsed)
# Lambda is derived from the half-life (in days).
DECAY_HALFLIFE = 30
DECAY_LAMBDA = np.log(2) / DECAY_HALFLIFE

# ==========================================
# Model / Inference Hyperparameters
# ==========================================

# Prediction Count
TOP_K = 12

# Graph Pruning
# Number of neighbors to retain per item in the similarity matrix
SIMILARITY_TOP_K = 100

# Stratification Offsets
# These offsets enforce a strict priority hierarchy:
# Habit (Repurchase) > Time-Decayed CF > Global Trend
# The gaps are large enough to prevent lower-tier signals from overriding higher tiers
# unless the higher tier has no candidates.
OFFSET_HABIT = 2000.0
OFFSET_CF = 100.0
OFFSET_TREND = 0.0

# ==========================================
# Compute Configuration
# ==========================================

# Precision
# Using float32 to prevent numerical instability with large stratification offsets
FLOAT_DTYPE = np.float32
