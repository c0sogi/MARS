import os
import torch
import random
import numpy as np

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

# Specific output directory for this experimental run (Idea 20)
OUTPUT_DIR = os.path.join(WORKING_DIR, "idea_20")

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VALIDATION_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
IMAGE_SIZE = 256
NUM_CHANNELS = 6  # 3 Channels (Ash Color) + 3 Channels (Band Differences)
NUM_WORKERS = 4  # Optimized for 12 vCPUs

# Temporal parameters (though dataset is pre-processed, these define the logic)
N_TIMES_BEFORE = 4
N_TIMES_AFTER = 3

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
BACKBONE = "convnext_tiny"
PRETRAINED = True
ENCODER_IN_CHANNELS = 3  # Standard ConvNeXt input
MODEL_INPUT_CHANNELS = 6  # Actual model input (modified first layer or adapter)

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
SEED = 42
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 10.0

# Loss Configuration
LOSS_BCE_WEIGHT = 1.0
LOSS_DICE_WEIGHT = 1.0

# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================
THRESHOLD = 0.5
USE_TTA = True  # Test Time Augmentation (Flip/Rotate)

# =============================================================================
# COMPUTE
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def setup_system(seed=SEED):
    """
    Creates necessary directories and sets random seeds for reproducibility.
    """
    # Create directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Set seeds
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Automatically setup on import
setup_system()
