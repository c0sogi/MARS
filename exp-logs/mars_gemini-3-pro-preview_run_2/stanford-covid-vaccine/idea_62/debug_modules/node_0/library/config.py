import os

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_62"

# Ensure working directory exists for cache and model artifacts
os.makedirs(WORKING_DIR, exist_ok=True)

# Raw Input Files
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")
SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata Files (Stratified Splits)
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Cache Files (Explicit Cache Invalidation keys)
CACHE_TRAIN = os.path.join(WORKING_DIR, "train_data_hs_gfn_v1.npz")
CACHE_VAL = os.path.join(WORKING_DIR, "val_data_hs_gfn_v1.npz")
CACHE_TEST = os.path.join(WORKING_DIR, "test_data_hs_gfn_v1.npz")

# Output Files
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

# =============================================================================
# DATA PARAMETERS
# =============================================================================
SEQ_LEN = 107
SCORED_LEN = 68

# Target Definitions
# All 5 targets present in the data
TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
# The 3 targets actually scored by the metric
SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
# The 2 targets that are unscored (to be masked in feedback)
UNSCORED_TARGETS = ["deg_pH10", "deg_50C"]
NUM_TARGETS = 5

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# Main Backbone (Dense Dilated TCN)
GROWTH_RATE = 64  # Capacity for the main encoder
LATENT_DIM = 64  # Dimension of the static latent representation Z
DILATIONS = [1, 2, 4, 8, 16, 32]  # Exponential dilation schedule
KERNEL_SIZE = 3
DROPOUT = 0.1

# Global-Context Feedback Module
FEEDBACK_GROWTH = 16  # Reduced capacity for the feedback encoder
FEEDBACK_OUT_DIM = 32  # Dimension of the feedback embedding E_fb

# Interaction & Aggregation
RNN_HIDDEN = 64  # Hidden size for the final Bidirectional GRU

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 16  # Strictly set to 16 (Lesson 00129)
LEARNING_RATE = 1e-3  # AdamW learning rate
EPOCHS = 50  # Max epochs (controlled by early stopping)
SEED = 42  # Fixed seed for reproducibility
NUM_WORKERS = 2  # DataLoader workers
