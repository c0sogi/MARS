import os

# =============================================================================
# Global Configuration & Reproducibility
# =============================================================================
SEED = 42

# =============================================================================
# File Paths & Directories
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"

# Cache Directory for Idea 31 (Deterministic Data Processing)
CACHE_DIR = os.path.join(WORKING_DIR, "idea_31")
os.makedirs(CACHE_DIR, exist_ok=True)

# Dataset Paths
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output Paths
SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
MODEL_SAVE_PATH = os.path.join(CACHE_DIR, "best_model.pth")

# =============================================================================
# Data Processing Parameters
# =============================================================================
TARGET_COL = "target"
ID_COL = "id"

# Feature Engineering
# f_27 is decomposed into 10 fixed-position character columns
NUM_CHAR_POSITIONS = 10

# =============================================================================
# Model Architecture: Frequency-Enhanced Parallel Funnel Ensemble (FEPFE)
# =============================================================================
# General
NUM_STREAMS = 5
EMBED_DIM = 16  # Independent embedding dim for each stream

# Stream Backbone Topologies (Hidden Layer Sizes)
# Streams 1, 2, 3: Standard Funnel (512 -> 256 -> 128)
# Streams 4, 5: Wide Funnel (1024 -> 512 -> 256)
STREAM_CONFIGS = [
    [512, 256, 128],  # Stream 1
    [512, 256, 128],  # Stream 2
    [512, 256, 128],  # Stream 3
    [1024, 512, 256],  # Stream 4
    [1024, 512, 256],  # Stream 5
]

# Regularization: Dropout Rates
# Varied between 0.20 and 0.30 across streams to enforce heterogeneity
DROPOUT_RATES = [0.20, 0.22, 0.25, 0.28, 0.30]

# Activation Function
ACTIVATION_FUNC = "ReLU"

# =============================================================================
# Training Hyperparameters
# =============================================================================
BATCH_SIZE = 1024
EPOCHS = 50

# Optimizer: AdamW
WEIGHT_DECAY = 2e-5

# Scheduler: OneCycleLR
MAX_LR = 1e-2
PCT_START = 0.3  # Default warm-up percentage for OneCycle
