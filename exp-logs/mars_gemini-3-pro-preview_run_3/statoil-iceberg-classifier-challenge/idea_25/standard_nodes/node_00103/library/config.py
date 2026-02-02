import os
import torch

# ==========================================
# Global Random Seed
# ==========================================
SEED = 42

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 75
PATIENCE = 12
NUM_FOLDS = 5

# Debugging / Development
DEBUG = False
MAX_DEBUG_SAMPLES = 100  # Used if DEBUG is True to limit dataset size

# ==========================================
# Data Dimensions & Configuration
# ==========================================
IMAGE_SIZE = 75
NUM_CHANNELS = 3  # HH, HV, and Average((HH, HV))

# ==========================================
# Compute Configuration
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# We have 12 vCPUs, so 4 workers is a safe and efficient default
NUM_WORKERS = 4

# ==========================================
# File Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Working Directory for Idea 25 (Max-Attentive Plain CNN)
WORKING_DIR = "./working/idea_25"
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

# Submission Paths
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")


# ==========================================
# Directory Setup Utility
# ==========================================
def setup_directories():
    """
    Ensures that the necessary working and output directories exist.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)


# Execute setup immediately when module is imported
setup_directories()
