import os
import torch
import random
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# ==========================================
# Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Working directory specific to this idea/run
WORKING_DIR = "./working/idea_41"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
MODELS_DIR = os.path.join(WORKING_DIR, "models")
SUBMISSION_PATH = "./submission/submission.csv"

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

# ==========================================
# Data Configuration
# ==========================================
IMG_SIZE = 224
NUM_CLASSES = 1  # Binary Classification (MGMT_value)

# Excluded cases as per task description
EXCLUDE_CASES = [109, 123, 709]

# Relative depths for the Spatially-Stratified Ensemble
# Cite Lesson 00018: A simple model focusing on the single most informative instance (center) often outperforms complex aggregation.
PLANES = {"center": 0.0}

# ==========================================
# Training Hyperparameters
# ==========================================
SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
DROPOUT_RATE = 0.3
N_FOLDS = 5

# Compute Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# Utilities
# ==========================================
def seed_everything(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# Augmentation Pipeline
# ==========================================
def get_transforms(phase="train"):
    """
    Returns the Albumentations transform pipeline for training or validation.

    Strategy:
    - Inputs are assumed to be 3-channel images (FLAIR, T1wCE, T2w) already
      cropped to the Brain ROI and resized to IMG_SIZE.
    - Inputs are assumed to be Min-Max scaled to [0, 1] per channel.
    - Training: Applies spatial augmentations (Flip, Rotate, Elastic, Grid).
      Strictly excludes Translation (Shift) and Scaling to preserve the
      spatial priors established by the ROI cropping.
    - Validation: Normalization and Tensor conversion only.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Rotation is allowed, but Shift/Scale are excluded
                A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
                # Non-rigid deformations
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    alpha_affine=50,
                    p=0.2,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.3,
                    p=0.2,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                # Normalize [0, 1] input to [-1, 1] for stable training
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                # Normalize [0, 1] input to [-1, 1]
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ]
        )
