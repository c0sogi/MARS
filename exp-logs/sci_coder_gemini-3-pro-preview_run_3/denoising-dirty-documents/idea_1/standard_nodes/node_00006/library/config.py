import os

# =============================================================================
# DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TRAIN_CLEANED_DIR = os.path.join(INPUT_DIR, "train_cleaned")
TEST_DIR = os.path.join(INPUT_DIR, "test")

METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_1"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# FILE PATHS
# =============================================================================
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sampleSubmission.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GENERAL CONFIGURATION
# =============================================================================
SEED = 42
PIXEL_MAX = 255.0

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# Patch size (k) for the k x k linear filter
# The model learns a linear combination of pixels in this window to predict the center.
PATCH_SIZE = 5
PATCH_AREA = PATCH_SIZE * PATCH_SIZE

# Regularization strength for Ridge Regression
# Helps prevent overfitting to specific noise artifacts.
ALPHA = 1.0

# Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10

# =============================================================================
# TRAINING & DATA PROCESSING CONFIGURATION
# =============================================================================
# Number of patches to randomly sample from the training set for solving the linear system.
# Using a subset is memory-efficient and sufficient for learning the kernel.
NUM_SAMPLES = 200000

# Batch size for inference (processing images one by one or in batches)
INFERENCE_BATCH_SIZE = 1

# =============================================================================
# CACHING PATHS
# =============================================================================
# Paths for caching processed numpy arrays to speed up subsequent runs
CACHED_X_PATH = os.path.join(WORKING_DIR, "X_train_patches.npy")
CACHED_Y_PATH = os.path.join(WORKING_DIR, "y_train_targets.npy")

# Paths for saving the learned model parameters
MODEL_WEIGHTS_PATH = os.path.join(WORKING_DIR, "model_weights.npy")
MODEL_BIAS_PATH = os.path.join(WORKING_DIR, "model_bias.npy")
