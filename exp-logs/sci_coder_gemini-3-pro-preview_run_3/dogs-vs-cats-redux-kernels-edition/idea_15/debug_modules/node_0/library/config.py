import os
import torch

# =============================================================================
# Directories and Paths
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_15"

# Ensure the working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

# =============================================================================
# Global Configuration
# =============================================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Compute constraints
# A100 has 40GB, so we can afford decent batch sizes.
# However, MaxViT and ConvNeXt at higher resolutions consume more memory.
# 32 is a safe conservative baseline for all models.
BATCH_SIZE = 32
NUM_WORKERS = 4  # 12 vCPUs available

# Debugging / Development
# Set DEBUG to True to run on a small subset of data for quick pipeline verification
DEBUG = False
SUBSET_SIZE = 100 if DEBUG else None

# =============================================================================
# Model Specifications (Heterogeneous Ensemble)
# =============================================================================
# Defines the specific hyperparameters for each model in the ensemble.
# Strategies implemented:
# 1. Resolution Diversity: 256 (ResNet), 288 (ConvNeXt), 224 (MaxViT)
# 2. Asynchronous Schedules: 8 epochs (CNNs), 15 epochs (Transformer)
MODEL_SPECS = {
    "resnet": {
        "timm_name": "resnet50.a1_in1k",
        "img_size": 256,
        "epochs": 8,
        "learning_rate": 1e-4,
        "weight_decay": 1e-2,
        "scheduler_min_lr": 1e-6,
    },
    "convnext": {
        "timm_name": "convnext_small.fb_in1k",
        "img_size": 288,
        "epochs": 8,
        "learning_rate": 1e-4,
        "weight_decay": 1e-2,
        "scheduler_min_lr": 1e-6,
    },
    "maxvit": {
        "timm_name": "maxvit_tiny_tf_224.in1k",
        "img_size": 224,
        "epochs": 15,  # Extended training for Transformer convergence
        "learning_rate": 5e-5,  # Lower LR for stability
        "weight_decay": 2e-2,  # Slightly higher regularization
        "scheduler_min_lr": 1e-6,
    },
}
