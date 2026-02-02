import os

# -----------------------------------------------------------------------------
# Global Configuration & Paths
# -----------------------------------------------------------------------------

# Base Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Working Directory for Idea 29 (Caching and Models)
WORKING_DIR = "./working/idea_29"
os.makedirs(WORKING_DIR, exist_ok=True)

# Submission Directory
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Model Save Path
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

# Caching Paths (Parquet/Numpy based on implementation needs)
# These paths are used by the data processing module to store/load processed tensors
TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_cache.npy")
TRAIN_LABEL_CACHE_PATH = os.path.join(WORKING_DIR, "train_labels.npy")
VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_cache.npy")
VAL_LABEL_CACHE_PATH = os.path.join(WORKING_DIR, "val_labels.npy")
TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_cache.npy")
# Auxiliary cache for ROI indices if needed
ROI_CACHE_PATH = os.path.join(WORKING_DIR, "roi_cache.parquet")

# -----------------------------------------------------------------------------
# Data Processing & Input Configuration
# -----------------------------------------------------------------------------

SEED = 42
IMG_SIZE = 224
NUM_SLICES_PER_GROUP = 3  # [Anchor-Stride, Anchor, Anchor+Stride]

# ROI Selection Parameters
ROI_DEPTH_RANGE = (0.15, 0.85)  # Restrict anchor search to 15%-85% depth
ROI_ANCHOR_MODALITY = "FLAIR"  # Use FLAIR Sum of Intensity for anchor

# Focal-Modality Input Configuration
# Defines the 12-channel input tensor structure (4 groups x 3 channels)
# Logic: Reallocate T1w capacity to T1wCE to capture both Context and Detail
CHANNEL_CONFIG = [
    # Group 1: FLAIR Context (Edema/Structure)
    {"modality": "FLAIR", "stride": 5, "group_idx": 0},
    # Group 2: T2w Context (Necrosis/Fluid)
    {"modality": "T2w", "stride": 5, "group_idx": 1},
    # Group 3: T1wCE Context (Enhancing Tumor Shape)
    {"modality": "T1wCE", "stride": 5, "group_idx": 2},
    # Group 4: T1wCE Detail (Fine-grained Texture)
    {"modality": "T1wCE", "stride": 2, "group_idx": 3},
]

NUM_GROUPS = len(CHANNEL_CONFIG)
TOTAL_INPUT_CHANNELS = NUM_GROUPS * NUM_SLICES_PER_GROUP  # 12 Channels

# -----------------------------------------------------------------------------
# Model Hyperparameters
# -----------------------------------------------------------------------------

BACKBONE_NAME = "efficientnet_b0"
PRETRAINED = True
DROP_RATE = 0.3  # Dropout rate for the classification head

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------

BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
EARLY_STOPPING_PATIENCE = 5

# Augmentation Parameters
ROTATION_DEGREES = 15  # +/- 15 degrees

# -----------------------------------------------------------------------------
# Compute Resources
# -----------------------------------------------------------------------------

NUM_WORKERS = 4
DEVICE = "cuda"  # NVIDIA A100 is available

# -----------------------------------------------------------------------------
# Debugging / Development
# -----------------------------------------------------------------------------

# Set to a small integer (e.g., 50) to run a quick test on a subset of data
# Set to None for full training
DEBUG_MAX_SAMPLES = None
