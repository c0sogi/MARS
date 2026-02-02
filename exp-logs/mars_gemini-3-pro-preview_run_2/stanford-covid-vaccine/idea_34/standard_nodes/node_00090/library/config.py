import os

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_34"

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Cache File Paths (for processed tensors)
# Using specific versioning to ensure cache invalidation if logic changes
TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data_lf_dcn_v1.npz")
VAL_CACHE = os.path.join(WORKING_DIR, "val_data_lf_dcn_v1.npz")
TEST_CACHE = os.path.join(WORKING_DIR, "test_data_lf_dcn_v1.npz")

# Output Paths
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_PATH = "./submission.csv"

# =============================================================================
# DATA SPECIFICATIONS
# =============================================================================
SEQ_LENGTH = 107
SCORED_LENGTH = 68

# Target Columns in order of Submission Format
TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
NUM_TARGETS = len(TARGET_COLS)

# Columns used for the MCRMSE metric
SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

# Indices of scored columns within the TARGET_COLS list
# reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
# Used to filter predictions/targets during loss calculation
SCORED_INDICES = [0, 1, 3]

# =============================================================================
# MODEL HYPERPARAMETERS (LF-DCN)
# =============================================================================
HIDDEN_DIM = 64  # Growth rate for DenseNet and Latent Dimension (Z)
FEEDBACK_DIM = 32  # Dimension for projected recycled predictions (E)
DROPOUT = 0.1
NUM_LAYERS = 6  # Number of dilated residual blocks
KERNEL_SIZE = 3
DILATIONS = [1, 2, 4, 8, 16, 32]  # Exponential dilation schedule matching NUM_LAYERS

# RNN Configuration
# Input dim to RNN is (HIDDEN_DIM + FEEDBACK_DIM) * 2 (Self + Partner) = 192
# Constraint: Hidden size is set to input_dim // 2 = 96
RNN_HIDDEN_DIM = 96
RNN_LAYERS = 1  # Standard BiGRU depth

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 16  # Adjusted for memory safety with dense connections
LEARNING_RATE = 1e-3
EPOCHS = 50
PATIENCE = 10  # For Early Stopping
NUM_WORKERS = 2  # CPU workers for dataloaders
SEED = 42  # Reproducibility

# Loss weights for iterative refinement
LOSS_FACTOR_FINAL = 1.0
LOSS_FACTOR_AUX = 0.5  # Weight for the first pass prediction
