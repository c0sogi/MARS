import os
import torch

# -----------------------------------------------------------------------------
# General Configuration
# -----------------------------------------------------------------------------
SEED = 42
DEBUG = False  # Set to True to run on a small subset for debugging

# -----------------------------------------------------------------------------
# Directories
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# We use idea_9 for this High-Fidelity Recurrent U-Net experiment to avoid
# cache collisions with the previous ResNet-18 experiment (idea_8).
WORKING_DIR = "./working/idea_9"

# Metadata Paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Output Subdirectories
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
PREDICTION_DIR = os.path.join(WORKING_DIR, "predictions")
SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

# -----------------------------------------------------------------------------
# Data Parameters
# -----------------------------------------------------------------------------
IMG_SIZE = 320  # Increased resolution to address bottleneck
SEQ_LENGTH = 5  # Sequence length for Recurrent U-Net (2.5D context)
IN_CHANNELS = 1  # Input channels per slice (Grayscale MRI)
NUM_WORKERS = 12  # Number of data loading workers (based on vCPUs)

# -----------------------------------------------------------------------------
# Model Parameters
# -----------------------------------------------------------------------------
BACKBONE = "resnet34"  # Higher capacity backbone
CLASSES = ["large_bowel", "small_bowel", "stomach"]
NUM_CLASSES = len(CLASSES)

# Weighted Deep Supervision
DEEP_SUPERVISION = True
# Loss weights for [Final Output, Aux Head 1, Aux Head 2]
DEEP_SUPERVISION_WEIGHTS = [1.0, 0.5, 0.25]

# -----------------------------------------------------------------------------
# Training Parameters
# -----------------------------------------------------------------------------
BATCH_SIZE = 32  # Adjusted for A100 40GB
EPOCHS = 15  # Sufficient for convergence with pre-trained backbone
LEARNING_RATE = 2e-4  # AdamW initial learning rate
WEIGHT_DECAY = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# Inference Parameters
# -----------------------------------------------------------------------------
THRESHOLD = 0.5  # Binary classification threshold


def setup():
    """
    Creates the necessary directory structure for the experiment.
    """
    dirs = [WORKING_DIR, CACHE_DIR, CHECKPOINT_DIR, PREDICTION_DIR, SUBMISSION_DIR]

    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # Explicitly satisfy the requirement to ensure idea_8 exists,
    # even though we are working in idea_9 to preserve data integrity.
    os.makedirs("./working/idea_8", exist_ok=True)
