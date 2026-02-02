import os
import torch

# =============================================================================
# DIRECTORY AND FILE PATHS
# =============================================================================

# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_72"

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# Raw Input Files (JSON)
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")
SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata Files (CSV - Stratified Splits)
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Cache Files (Numpy Archives)
# Using specific version v1 for HC-HIDN architecture
CACHE_TRAIN = os.path.join(WORKING_DIR, "train_data_hc_hidn_v1.npz")
CACHE_VAL = os.path.join(WORKING_DIR, "val_data_hc_hidn_v1.npz")
CACHE_TEST = os.path.join(WORKING_DIR, "test_data_hc_hidn_v1.npz")

# Output Files
MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

# =============================================================================
# DATA SPECIFICATIONS
# =============================================================================

SEQ_LENGTH = 107
SEQ_SCORED = 68

# Target Columns in the dataset
TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
NUM_TARGETS = len(TARGET_COLS)

# Indices of targets that are actually scored in the metric
# reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
SCORED_TARGET_INDICES = [0, 1, 3]

# Vocabulary Sizes
VOCAB_SIZE = 4  # A, G, C, U
STRUCTURE_VOCAB_SIZE = 3  # ., (, )
LOOP_VOCAB_SIZE = 7  # S, M, I, B, H, E, X

# =============================================================================
# MODEL HYPERPARAMETERS (HC-HIDN)
# =============================================================================

# Backbone Capacity
HIDDEN_DIM = 64  # Compact hidden size for RNN/Interaction
GROWTH_RATE = 64  # High capacity for DenseNet backbone
DROPOUT = 0.1

# Architecture Specifics
USE_HYBRID_STEM = True
KERNEL_SIZE = 3
DILATIONS = [1, 2, 4, 8, 16, 32]  # Exponential dilation for receptive field

# Feedback Mechanism
FEEDBACK_DIM = 32  # Dimension for feedback embeddings
RECYCLING_STEPS = 2  # Total passes: Pass 1 (init), Pass 2 (feedback)

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

BATCH_SIZE = 16  # Strictly 16 for gradient frequency
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
PATIENCE = 5  # Early stopping patience
MAX_GRAD_NORM = 1.0

# Loss weighting
# Total Loss = MCRMSE(Pass2) + AUX_WEIGHT * MCRMSE(Pass1)
AUX_WEIGHT = 0.5

# =============================================================================
# REPRODUCIBILITY & DEBUGGING
# =============================================================================

SEED = 42
DEBUG = False  # Set to True to run on a small subset
DEBUG_SUBSET_SIZE = 50  # Number of samples to use in debug mode
NUM_WORKERS = 2  # Data loader workers

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_config_dict():
    """Returns all configuration constants as a dictionary."""
    return {k: v for k, v in globals().items() if k.isupper()}
