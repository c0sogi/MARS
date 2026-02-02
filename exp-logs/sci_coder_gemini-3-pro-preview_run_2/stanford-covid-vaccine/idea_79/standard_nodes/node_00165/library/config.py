import os

# =============================================================================
# GENERAL SETTINGS
# =============================================================================
SEED = 42
NUM_WORKERS = 2  # Optimized for the available vCPUs

# =============================================================================
# PATHS AND DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

# Metadata Files (Pre-generated)
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

# Experiment Directory (Idea 79: RIS-DRN)
EXPERIMENT_ID = "idea_79"
CACHE_DIR = os.path.join(WORKING_DIR, EXPERIMENT_ID)

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Cache Files (Explicit Cache Invalidation Keys)
TRAIN_CACHE_KEY = "train_data_ris_drn_v1.npz"
VAL_CACHE_KEY = "val_data_ris_drn_v1.npz"
TEST_CACHE_KEY = "test_data_ris_drn_v1.npz"

TRAIN_CACHE_PATH = os.path.join(CACHE_DIR, TRAIN_CACHE_KEY)
VAL_CACHE_PATH = os.path.join(CACHE_DIR, VAL_CACHE_KEY)
TEST_CACHE_PATH = os.path.join(CACHE_DIR, TEST_CACHE_KEY)

# Model Artifacts
BEST_MODEL_PATH = os.path.join(CACHE_DIR, "best_model.pth")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# DATASET SPECIFICATIONS
# =============================================================================
SEQ_LENGTH = 107
SCORED_SEQ_LENGTH = 68

# Full list of targets in the training data
ALL_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

# Targets strictly used for scoring and loss calculation
SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

# Indices of SCORED_COLS within ALL_TARGETS (0, 1, 3)
SCORED_INDICES = [i for i, col in enumerate(ALL_TARGETS) if col in SCORED_COLS]

# =============================================================================
# MODEL HYPERPARAMETERS (RIS-DRN)
# =============================================================================
# Main Backbone (Raw-Injecting Dense Dilated TCN)
LATENT_DIM = 64
GROWTH_RATE = 64
DILATION_RATES = [1, 2, 4, 8, 16, 32]
KERNEL_SIZE_STEM = 3
KERNEL_SIZE_POINTWISE = 1
DROPOUT = 0.1

# Feedback Module
FEEDBACK_DIM = 32
FEEDBACK_GROWTH_RATE = 16

# Interaction & Aggregation
RNN_HIDDEN_DIM = 64
BIDIRECTIONAL = True

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 16  # Strictly set to 16 for gradient frequency
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Scheduler
LR_FACTOR = 0.5
LR_PATIENCE = 5

# Loss Weights for Iterative Refinement
LOSS_WEIGHT_PASS_2 = 1.0
LOSS_WEIGHT_PASS_1 = 0.5
