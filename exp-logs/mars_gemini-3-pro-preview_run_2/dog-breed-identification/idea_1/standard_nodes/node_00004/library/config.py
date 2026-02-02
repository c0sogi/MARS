import os
import torch

# ==========================================
# Random Seed & Reproducibility
# ==========================================
SEED = 42

# ==========================================
# Hardware Configuration
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Number of workers for data loading (adjust based on vCPUs)
NUM_WORKERS = 4

# ==========================================
# Directory & File Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_1"
SUBMISSION_DIR = "./submission"

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output Files
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Cache Files for Embeddings (using .npy as requested)
TRAIN_EMBEDDINGS_PATH = os.path.join(WORKING_DIR, "train_embeddings.npy")
TRAIN_LABELS_PATH = os.path.join(WORKING_DIR, "train_labels.npy")
VAL_EMBEDDINGS_PATH = os.path.join(WORKING_DIR, "val_embeddings.npy")
VAL_LABELS_PATH = os.path.join(WORKING_DIR, "val_labels.npy")
TEST_EMBEDDINGS_PATH = os.path.join(WORKING_DIR, "test_embeddings.npy")
TEST_IDS_PATH = os.path.join(WORKING_DIR, "test_ids.npy")

# ==========================================
# Data Configuration
# ==========================================
IMG_SIZE = 224
CHANNELS = 3
NUM_CLASSES = 120
BATCH_SIZE = 64

# ImageNet Normalization Statistics
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ==========================================
# Model Configuration (ResNet50 + LogReg)
# ==========================================
MODEL_NAME = "resnet50"
# ResNet50 Global Average Pooling output dimension
EMBEDDING_DIM = 2048

# Logistic Regression Hyperparameters
LR_SOLVER = "lbfgs"
LR_MAX_ITER = 1000
LR_C = 1.0  # Inverse of regularization strength (smaller = stronger regularization)
LR_MULTI_CLASS = "multinomial"
LR_N_JOBS = -1  # Use all available cores for the sklearn solver

# ==========================================
# Deep Learning Training Configuration
# ==========================================
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
