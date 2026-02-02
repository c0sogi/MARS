import os
import torch

# ==========================================
# Reproducibility
# ==========================================
SEED = 42

# ==========================================
# Paths
# ==========================================
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")

METADATA_DIR = "./metadata"
TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
VAL_META = os.path.join(METADATA_DIR, "val.csv")
TEST_META = os.path.join(METADATA_DIR, "test.csv")

# Working Directory for Idea 3
WORKING_DIR = "./working/idea_3"
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Compute
# ==========================================
# 12 vCPUs available, 4 is a safe number for dataloader workers
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# Model Configuration
# ==========================================
MODEL_NAME = "convnext_tiny"
NUM_CLASSES = 1  # Binary classification: Dog (1) vs Cat (0)
PRETRAINED = True

# ==========================================
# Data Configuration
# ==========================================
IMG_SIZE = 224
BATCH_SIZE = 64  # A100 GPU can handle larger batches, 64 is efficient for Tiny

# ==========================================
# Training Configuration
# ==========================================
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2  # Standard for AdamW

# Mixup Regularization
# Alpha for Beta distribution. 0.2 is a common starting point for ImageNet-like tasks.
MIXUP_ALPHA = 0.2

# Scheduler (Cosine Annealing)
ETA_MIN = 1e-6

# ==========================================
# Debugging / Development
# ==========================================
# If set to an integer (e.g., 1000), the dataloader/training loop should
# subset the data to this amount for rapid debugging.
DEBUG_SUBSET_SIZE = None
