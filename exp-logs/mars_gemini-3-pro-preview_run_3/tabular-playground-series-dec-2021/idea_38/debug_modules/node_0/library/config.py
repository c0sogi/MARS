import os
import torch

# =============================================================================
# Paths & Directories
# =============================================================================
# Input Data (Metadata)
METADATA_DIR = "./metadata"
TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

# Output & Cache
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_38")
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure writeable directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# Hardware & Reproducibility
# =============================================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4

# CuDNN Configuration
# Disable strict determinism to maximize kernel performance (Lesson 00070)
CUDNN_DETERMINISTIC = False
CUDNN_BENCHMARK = True

# =============================================================================
# Data Configuration
# =============================================================================
TARGET_COL = "Cover_Type"
ID_COL = "Id"

# Class Configuration
# The dataset contains classes 1-7. We set NUM_CLASSES to 8 to handle 1-based indexing
# comfortably (index 0 unused), or 7 if mapping 1-7 -> 0-6.
# Standard practice for this dataset is usually mapping to 0-6.
NUM_CLASSES = 7

# Debugging / Development
# Set to an integer (e.g., 10000) to limit dataset size for fast prototyping
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None

# =============================================================================
# Model Architecture: Wide Asymmetric Parallel Vector-DCN-ResNet
# =============================================================================
# Branch 1: Vector-Based DCN (Rank-1 Cross Layers)
DCN_LAYERS = 3  # Asymmetric depth (Lesson 00071)
DCN_INIT_STD = 1e-4  # Near-zero initialization (Lesson 00066)

# Branch 2: Wide Full Pre-Activation ResNet
RESNET_BLOCKS = 4  # Optimal depth for convergence (Lesson 00031)
HIDDEN_DIM = 1024  # Scaled width (Lesson 00029, 00054)
DROPOUT_RATE = 0.3  # Increased regularization for wider model (Lesson 00056)

# =============================================================================
# Training Hyperparameters
# =============================================================================
BATCH_SIZE = 4096  # High batch size for budget efficiency
EPOCHS = 60  # Fixed budget (Lesson 00022)

# Optimizer (AdamW)
LEARNING_RATE = 1e-3  # Standard base LR
WEIGHT_DECAY = 1e-2  # Decoupled weight decay

# Scheduler (ReduceLROnPlateau)
SCHEDULER_FACTOR = 0.1  # Aggressive decay (Lesson 00068)
SCHEDULER_PATIENCE = 5  # Patience before decay
SCHEDULER_MODE = "max"  # Monitor validation accuracy

# Early Stopping
EARLY_STOPPING_PATIENCE = 12  # Stop after stagnation
