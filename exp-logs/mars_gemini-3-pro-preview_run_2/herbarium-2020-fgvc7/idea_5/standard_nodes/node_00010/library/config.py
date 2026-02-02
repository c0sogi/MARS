import os
import torch
import random
import numpy as np

# =============================================================================
# DIRECTORY PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_5"
SUBMISSION_DIR = "./submission"

# Ensure working and output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# FILE PATHS
# =============================================================================
# Generated metadata CSVs
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Raw metadata JSON for taxonomy extraction (Family/Genus mapping)
RAW_TRAIN_METADATA_PATH = os.path.join(INPUT_DIR, "nybg2020/train/metadata.json")

# Sample submission file
SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

# Caching paths for deterministic data processing
TAXONOMY_MAPPING_PATH = os.path.join(WORKING_DIR, "taxonomy_mapping.parquet")
LABEL_ENCODERS_PATH = os.path.join(WORKING_DIR, "label_encoders.npy")
BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
IMAGE_SIZE = 300
NUM_SPECIES_CLASSES = 32093
# Note: Genus and Family class counts are determined dynamically from the taxonomy mapping

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
MODEL_NAME = "efficientnet_b3"
PRETRAINED = True
DROPOUT_RATE = 0.3
EMBEDDING_SIZE = 1536  # EfficientNet-B3 final feature map size

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 15
WEIGHT_DECAY = 1e-4
SEED = 42

# Multi-Task Loss Weights
# Balancing the contribution of fine-grained (species) vs coarse-grained (family/genus) tasks
LOSS_WEIGHTS = {"species": 1.0, "genus": 0.5, "family": 0.5}

# Focal Loss Hyperparameters (for Species Head)
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

# Scheduler Settings (Cosine Annealing)
SCHEDULER_T_MAX = NUM_EPOCHS
SCHEDULER_MIN_LR = 1e-6

# Early Stopping
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.001

# =============================================================================
# HARDWARE CONFIGURATION
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 12
PIN_MEMORY = True


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
