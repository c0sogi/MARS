import os
import torch

# ==========================================
# Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_5"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Files
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# ==========================================
# Global Configuration
# ==========================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 12  # Optimized for 12 vCPUs

# ==========================================
# Data Configuration
# ==========================================
NUM_CLASSES = 120
BATCH_SIZE = 64  # Batch size for feature extraction (A100 40GB)

# Multi-View Configuration
# View 1: Standard (Resize -> CenterCrop)
VIEW_STANDARD_RESIZE = 256
VIEW_STANDARD_CROP = 224

# View 2: Global (Resize Squish)
VIEW_GLOBAL_SIZE = 224

# View 3: Local (Resize Large -> CenterCrop)
VIEW_LOCAL_RESIZE = 320
VIEW_LOCAL_CROP = 224

# ==========================================
# Model Configuration
# ==========================================
# Stream A: ConvNeXt-Large (CNN)
MODEL_A_NAME = "convnext_large"
MODEL_A_WEIGHTS = "IMAGENET1K_V1"
MODEL_A_EMBED_DIM = 1536
MODEL_A_TOTAL_DIM = MODEL_A_EMBED_DIM * 3  # 3 Views concatenated

# Stream B: ViT-Large-16 (Transformer)
MODEL_B_NAME = "vit_l_16"
MODEL_B_WEIGHTS = "IMAGENET1K_SWAG_E2E_V1"
MODEL_B_EMBED_DIM = 1024
MODEL_B_TOTAL_DIM = MODEL_B_EMBED_DIM * 3  # 3 Views concatenated

# ==========================================
# Training / Solver Configuration
# ==========================================
# Logistic Regression Hyperparameters
LOGREG_C = 1.0
LOGREG_MAX_ITER = 1000
LOGREG_SOLVER = "lbfgs"
LOGREG_MULTI_CLASS = "multinomial"

# ==========================================
# Debugging
# ==========================================
DEBUG = False
DEBUG_DATASET_SIZE = 100  # Number of samples to use when DEBUG is True
