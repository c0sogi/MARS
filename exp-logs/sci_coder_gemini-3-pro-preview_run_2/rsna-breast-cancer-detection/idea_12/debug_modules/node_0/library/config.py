import os
import torch

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_12"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Specific file paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache paths for processed data
CACHE_DIR = WORKING_DIR

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
SEED = 42
NUM_WORKERS = 4  # Number of subprocesses for data loading

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
# Image Input
IMAGE_SIZE = (640, 640)  # Higher resolution for better sensitivity
NUM_CHANNELS = 3  # 3 Channels: Grayscale, CLAHE, Gamma

# Tabular Input
# Features used in the tabular branch (must be present in test.csv)
NUMERICAL_COLS = ["age"]
CATEGORICAL_COLS = ["site_id", "laterality", "view", "implant", "machine_id"]

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Backbone
MODEL_NAME = "tf_efficientnetv2_s"  # EfficientNetV2-Small
PRETRAINED = True

# Hybrid Architecture Settings
TABULAR_EMBED_DIM = 64  # Dimension for categorical embeddings
TABULAR_HIDDEN_DIM = 128  # Hidden layer size for tabular MLP
MODALITY_DROPOUT_PROB = (
    0.5  # Probability of zeroing out tabular features during training
)

# Regularization
DROP_RATE = 0.3  # Head dropout
DROP_PATH_RATE = 0.2  # Stochastic depth

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
BATCH_SIZE = 24  # Fits 640x640 on A100-40GB
EPOCHS = 5  # Sufficient for fine-tuning
LEARNING_RATE = 1e-4  # Initial LR
WEIGHT_DECAY = 1e-2

# Loss Function
# High positive weight to counter 1:50 imbalance
POS_WEIGHT = 20.0
# Force loss calculation in Float32 to prevent NaN with high weights in AMP
USE_FP32_LOSS = True

# Debugging
DEBUG = False  # Set to True to train on a small subset
DEBUG_SAMPLE_SIZE = 1000

# =============================================================================
# HARDWARE
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
