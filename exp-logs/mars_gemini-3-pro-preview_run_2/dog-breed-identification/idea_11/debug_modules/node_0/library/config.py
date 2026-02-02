import os
import torch
import numpy as np
import random

# ==========================================
# Reproducibility
# ==========================================
SEED = 42


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ==========================================
# Directories and Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_11"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# ==========================================
# Model Hyperparameters
# ==========================================
# Backbone Architecture: ConvNeXt-Large
# Weights: torchvision "New Recipe" (IMAGENET1K_V1)
MODEL_NAME = "convnext_large"
MODEL_WEIGHTS = "IMAGENET1K_V1"

# Target Information
NUM_CLASSES = 120

# ==========================================
# Data Processing / Multi-View Config
# ==========================================
# The input size expected by the model backbone
INPUT_SIZE = 224

# View 1: Global View (Shape)
# Squish the image to 224x224, ignoring aspect ratio
GLOBAL_VIEW_RESIZE = (224, 224)

# View 2: Standard View (Context)
# Resize small edge to 232, then Center Crop to 224
STANDARD_VIEW_RESIZE = 232
STANDARD_VIEW_CROP = 224

# View 3: Robust Local View (Texture & Spatial Aggregation)
# Resize small edge to 288, then extract Five Crops (4 corners + center) of 224
LOCAL_VIEW_RESIZE = 288
LOCAL_VIEW_CROP = 224

# DataLoader Settings
BATCH_SIZE = 32
NUM_WORKERS = 4  # Optimized for 12 vCPUs

# ==========================================
# Caching & Storage
# ==========================================
# Paths for caching intermediate embeddings to ensure directory safety and speed
CACHE_PATHS = {
    "train_embeddings": os.path.join(WORKING_DIR, "train_embeddings.npy"),
    "train_labels": os.path.join(WORKING_DIR, "train_labels.npy"),
    "val_embeddings": os.path.join(WORKING_DIR, "val_embeddings.npy"),
    "val_labels": os.path.join(WORKING_DIR, "val_labels.npy"),
    "test_embeddings": os.path.join(WORKING_DIR, "test_embeddings.npy"),
    "test_ids": os.path.join(WORKING_DIR, "test_ids.npy"),
    "model": os.path.join(WORKING_DIR, "logreg_model.joblib"),
    "submission": os.path.join(SUBMISSION_DIR, "submission.csv"),
}

# ==========================================
# Classifier Configuration
# ==========================================
# LogisticRegressionCV settings
LOGREG_MAX_ITER = 1000
LOGREG_CV = 5
LOGREG_SOLVER = "lbfgs"
LOGREG_N_JOBS = -1  # Use all available cores
