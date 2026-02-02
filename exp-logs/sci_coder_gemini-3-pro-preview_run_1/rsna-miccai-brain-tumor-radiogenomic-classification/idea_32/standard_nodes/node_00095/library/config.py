import os
import torch

# ==========================================
# Global Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_33"
SUBMISSION_DIR = "./submission"

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Caching Configuration
# ==========================================
# Paths for caching processed numpy arrays to speed up training/inference
CACHE_TRAIN_IMAGES = os.path.join(WORKING_DIR, "train_images.npy")
CACHE_TRAIN_IDS = os.path.join(WORKING_DIR, "train_ids.npy")
CACHE_TRAIN_LABELS = os.path.join(WORKING_DIR, "train_labels.npy")

CACHE_VAL_IMAGES = os.path.join(WORKING_DIR, "val_images.npy")
CACHE_VAL_IDS = os.path.join(WORKING_DIR, "val_ids.npy")
CACHE_VAL_LABELS = os.path.join(WORKING_DIR, "val_labels.npy")

CACHE_TEST_IMAGES = os.path.join(WORKING_DIR, "test_images.npy")
CACHE_TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")

# ==========================================
# Data Parameters
# ==========================================
IMG_SIZE = 224
NUM_SLICES = 3  # Number of depth slices to extract
RELATIVE_DEPTHS = [0.4, 0.5, 0.6]  # Relative depths in the Brain ROI
MODALITIES = ["FLAIR", "T1wCE", "T2w"]
CHANNELS_PER_SLICE = len(MODALITIES)
# Total Input Channels = 3 slices * 3 modalities = 9 channels
INPUT_CHANNELS = NUM_SLICES * CHANNELS_PER_SLICE

# ==========================================
# Model Parameters
# ==========================================
BACKBONE = "efficientnet_b0"
NUM_CLASSES = 1
DROPOUT_RATE = 0.3  # Classifier dropout
INPUT_DROPOUT_PROB = 0.2  # Structured input dropout (triplet dropout)

# ==========================================
# Training Parameters
# ==========================================
SEED = 42
NUM_FOLDS = 5
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# Debugging / Development
# ==========================================
# Set DEBUG to True to run on a small subset of data for quick pipeline verification
DEBUG = False
MAX_DEBUG_SAMPLES = 50


# ==========================================
# Utility Functions
# ==========================================
def seed_everything(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    import random
    import numpy as np

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
