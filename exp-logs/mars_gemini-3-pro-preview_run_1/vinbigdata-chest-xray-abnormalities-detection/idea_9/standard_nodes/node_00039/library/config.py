import os
import torch

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_9"

# Metadata Files (Pre-generated)
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_meta.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val_meta.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test_meta.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output Directories
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
# Reproducibility
SEED = 42

# Data
IMG_SIZE = 640
NUM_CLASSES = 14  # Classes 0-13 (Findings). Class 14 (No finding) is handled via global head/absence.
NUM_WORKERS = 12

# Training
BATCH_SIZE = 8  # Adjusted for 16GB GPU (Cite debug_lesson_7)
EPOCHS = 20  # Extended training for strong augmentations
LEARNING_RATE = 1e-4  # AdamW default start
MAX_GRAD_NORM = 1.0  # Gradient clipping for stability

# Model
BACKBONE = "efficientnet_b0"

# Compute
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def setup_directories():
    """
    Creates the necessary working directory structure for the project.
    Includes subdirectories for caching processed arrays and saving checkpoints.
    """
    directories = [
        WORKING_DIR,
        CACHE_DIR,
        CHECKPOINT_DIR,
        SUBMISSION_DIR,
        os.path.join(CACHE_DIR, "train"),
        os.path.join(CACHE_DIR, "val"),
        os.path.join(CACHE_DIR, "test"),
    ]

    for d in directories:
        os.makedirs(d, exist_ok=True)

    print(f"Directory structure initialized at: {WORKING_DIR}")
