import os
import torch

# ==========================================
# Global Constants & Reproducibility
# ==========================================
SEED = 42

# ==========================================
# Compute Resources
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Adjust number of workers based on available vCPUs (12 available)
NUM_WORKERS = 4

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 1024
EPOCHS = 50
MAX_LR = 1e-2
WEIGHT_DECAY = 1e-5

# ==========================================
# Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Data Paths (using metadata splits)
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

# Working Directory for Caching (Parquet/Numpy files)
WORKING_DIR = "./working/idea_30"
os.makedirs(WORKING_DIR, exist_ok=True)
CACHE_DIR = WORKING_DIR

# Submission Directory
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Model Architecture: Multi-Resolution Parallel Funnel Ensemble (MRPFE)
# ==========================================
# Defines the configuration for the 5 independent streams
STREAMS_CONFIG = [
    # Stream 1 (Anchor): Standard Funnel, Emb 16, Drop 0.20
    {
        "name": "stream_1_anchor",
        "emb_dim": 16,
        "hidden_layers": [512, 256, 128],
        "dropout": 0.20,
    },
    # Stream 2 (Anchor): Standard Funnel, Emb 16, Drop 0.20
    {
        "name": "stream_2_anchor",
        "emb_dim": 16,
        "hidden_layers": [512, 256, 128],
        "dropout": 0.20,
    },
    # Stream 3 (High-Res): Wide Funnel, Emb 32, Drop 0.25
    {
        "name": "stream_3_high_res",
        "emb_dim": 32,
        "hidden_layers": [1024, 512, 256],
        "dropout": 0.25,
    },
    # Stream 4 (Low-Res): Standard Funnel, Emb 8, Drop 0.15
    {
        "name": "stream_4_low_res",
        "emb_dim": 8,
        "hidden_layers": [512, 256, 128],
        "dropout": 0.15,
    },
    # Stream 5 (Conservative): Standard Funnel, Emb 16, Drop 0.30
    {
        "name": "stream_5_conservative",
        "emb_dim": 16,
        "hidden_layers": [512, 256, 128],
        "dropout": 0.30,
    },
]

# ==========================================
# Feature Engineering Configuration
# ==========================================

# Raw Continuous Features: f_00 to f_28, excluding f_27
CONTINUOUS_FEATURES = [f"f_{i:02d}" for i in range(29) if i != 27]

# Raw Categorical Features
CATEGORICAL_FEATURES = ["f_29", "f_30"]

# String Feature to be decomposed
STRING_FEATURE = "f_27"
F27_SEQ_LEN = 10  # f_27 contains 10 characters

# Derived Features
FEATURE_UNIQUE_COUNT = "unique_character_count"

# List of all continuous features after engineering (for scaling)
ALL_CONTINUOUS_FEATURES = CONTINUOUS_FEATURES + [FEATURE_UNIQUE_COUNT]
