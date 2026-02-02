import os
import torch

# ==========================================
# General Configuration
# ==========================================
SEED = 42
DEBUG = False
DEBUG_SAMPLE_SIZE = 100  # Number of samples to use when DEBUG is True

# ==========================================
# Directories and Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_2"
SUBMISSION_DIR = "./submission"

# Create working and submission directories if they don't exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Output Paths
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "model_best.pth")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Model Hyperparameters
# ==========================================
MODEL_NAME = "tf_efficientnet_b5_ns"  # EfficientNet-B5 with Noisy Student weights
IMG_SIZE = 456  # Increased resolution for fine-grained features
NUM_CLASSES = 1  # Single neuron for regression
USE_GEM_POOLING = True  # Use Generalized Mean Pooling
DROPOUT_RATE = 0.3
DROP_PATH_RATE = 0.2

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 8  # Reduced to fit within ~16GB VRAM
EPOCHS = 15  # Sufficient for transfer learning convergence
LEARNING_RATE = 3e-4  # Initial learning rate for Adam
WEIGHT_DECAY = 1e-5  # Regularization
MAX_GRAD_NORM = 5.0  # Gradient clipping
PATIENCE = 4  # Early stopping patience
NUM_FOLDS = 5  # Number of folds for Cross Validation

# Scheduler (Cosine Annealing)
T_MAX = EPOCHS  # Cycle length
MIN_LR = 1e-6  # Minimum learning rate

# ==========================================
# Hardware Configuration
# ==========================================
NUM_WORKERS = 12  # Matches available vCPUs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
