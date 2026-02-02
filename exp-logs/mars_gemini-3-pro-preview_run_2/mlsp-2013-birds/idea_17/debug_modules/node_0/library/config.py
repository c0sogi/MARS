import os
import torch

# =============================================================================
# PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Working directory for this specific experiment (Idea 17)
WORKING_DIR = "./working/idea_17"

# Source of images: Filtered Spectrograms as per strategy
SPECTROGRAM_DIR = os.path.join(INPUT_DIR, "supplemental_data", "filtered_spectrograms")

# Metadata CSV paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Submission paths
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
SEED = 42
NUM_SPECIES = 19
N_FOLDS = 5
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 32
EPOCHS = 50  # Maximum epochs; Early Stopping should be handled by the trainer
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Augmentation
MIXUP_ALPHA = 0.4

# Distillation Parameters
# Gamma controls the balance between Hard Labels (Ground Truth) and Soft Labels (Teacher OOF).
# Loss = Gamma * Hard_Loss + (1 - Gamma) * Soft_Loss
DISTILLATION_GAMMA = 0.5

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================
# Defines the input resolution (Freq x Time) for each backbone in the heterogeneous ensemble.
MODEL_CONFIGS = {
    "resnet18": {"img_size": (224, 448)},
    "efficientnet_b0": {"img_size": (224, 448)},
    "densenet121": {"img_size": (160, 320)},
}

# =============================================================================
# DEBUGGING & CONTROL
# =============================================================================
# If set to an integer (e.g., 50), the data loader will only load this many samples.
# Useful for verifying pipeline mechanics without full training.
MAX_DEBUG_SAMPLES = None


# =============================================================================
# INITIALIZATION
# =============================================================================
def setup_directories():
    """Creates necessary working and submission directories."""
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)


# Execute setup on module import
setup_directories()
