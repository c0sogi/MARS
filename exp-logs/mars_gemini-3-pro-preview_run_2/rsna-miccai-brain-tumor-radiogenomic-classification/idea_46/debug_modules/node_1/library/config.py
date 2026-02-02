import os

# -----------------------------------------------------------------------------
# Global Configuration
# -----------------------------------------------------------------------------
SEED = 42
IDEA_NAME = "idea_46"

# -----------------------------------------------------------------------------
# Directory Paths
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
LABELS_FILE = os.path.join(INPUT_DIR, "train_labels.csv")
SAMPLE_SUBMISSION_FILE = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata Paths
METADATA_DIR = "./metadata"
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Working and Cache Directories
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, IDEA_NAME)
os.makedirs(CACHE_DIR, exist_ok=True)

# Submission Directory
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# Model Save Path
MODEL_SAVE_PATH = os.path.join(CACHE_DIR, "best_model.pth")

# -----------------------------------------------------------------------------
# Data Hyperparameters
# -----------------------------------------------------------------------------
IMG_SIZE = 224
MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]
NUM_SLICES = 5  # Number of slices per modality
STRIDE = 2  # Stride for slice selection (Anchor-4, -2, 0, +2, +4)
ROI_DEPTH_RANGE = (0.15, 0.85)  # Normalized depth range to search for ROI
EXCLUDE_IDS = [109, 123, 709]  # Problematic cases to exclude

# -----------------------------------------------------------------------------
# Model Hyperparameters
# -----------------------------------------------------------------------------
BACKBONE = "efficientnet_b0"
INPUT_CHANNELS = 20  # 4 modalities * 5 slices
GROUPS = 4  # Grouped convolution for the first layer (modality isolation)
DROPOUT = 0.5  # Classification head dropout probability
NUM_CLASSES = 1

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-2
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
NUM_WORKERS = 4  # Number of data loading workers

# -----------------------------------------------------------------------------
# Augmentation Hyperparameters
# -----------------------------------------------------------------------------
AUG_ROTATION_RANGE = 15  # Degrees (+/-) for random rotation
