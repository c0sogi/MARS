import os
import torch
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# Global Paths & Directories
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_19"

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata Files
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Submission Output
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# -----------------------------------------------------------------------------
# Global Hyperparameters
# -----------------------------------------------------------------------------
SEED = 42
N_FOLDS = 5
NUM_WORKERS = 4  # Optimized for 12 vCPUs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Quality Gating
OOF_THRESHOLD = 0.05  # Max allowed Log Loss to include model in ensemble


# -----------------------------------------------------------------------------
# Model Configuration Structure
# -----------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """
    Defines the hyperparameter structure for a specific model architecture.
    """

    model_name: str  # timm model name
    weights: str  # Pretrained weights identifier
    img_size: int  # Input resolution (height/width)
    epochs: int  # Training duration
    batch_size: int  # Batch size per GPU
    learning_rate: float  # Initial learning rate
    min_lr: float  # Minimum learning rate for cosine schedule
    weight_decay: float  # Optimizer weight decay


# -----------------------------------------------------------------------------
# Architecture Specific Configurations
# -----------------------------------------------------------------------------

# 1. ResNet-50: The Standard CNN Anchor
# Resolution: 256x256 (Standard receptive field balance)
# Schedule: 8 Epochs
RESNET_CONFIG = ModelConfig(
    model_name="resnet50.a1_in1k",
    weights="resnet50.a1_in1k",
    img_size=256,
    epochs=8,
    batch_size=128,  # A100 40GB allows large batch
    learning_rate=1e-4,
    min_lr=1e-6,
    weight_decay=1e-4,
)

# 2. ConvNeXt-Small: The Modern CNN
# Resolution: 288x288 (Fine-grained detail focus)
# Schedule: 8 Epochs
CONVNEXT_CONFIG = ModelConfig(
    model_name="convnext_small.fb_in1k",
    weights="convnext_small.fb_in1k",
    img_size=288,
    epochs=8,
    batch_size=64,
    learning_rate=1e-4,
    min_lr=1e-6,
    weight_decay=1e-4,
)

# 3. MaxViT-Tiny: The Multi-Axis Transformer
# Resolution: 224x224 (Native grid match)
# Schedule: 15 Epochs (Extended for attention convergence)
MAXVIT_CONFIG = ModelConfig(
    model_name="maxvit_tiny_tf_224.in1k",
    weights="maxvit_tiny_tf_224.in1k",
    img_size=224,
    epochs=15,
    batch_size=64,
    learning_rate=1e-4,
    min_lr=1e-6,
    weight_decay=1e-4,
)

# List of all configs for iteration
MODEL_CONFIGS = [RESNET_CONFIG, CONVNEXT_CONFIG, MAXVIT_CONFIG]


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def seed_everything(seed: int = SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and Torch.
    """
    import random
    import numpy as np

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
