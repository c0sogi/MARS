import os
import torch

# ==========================================
# Directories and File Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Using idea_2 directory for caching intermediate files as per requirements
WORKING_DIR = "./working/idea_2"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Raw Data Paths
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")
SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata Paths
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Output Paths
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_resnet18_model.pth")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Data Processing Hyperparameters
# ==========================================
# ResNet-18 expects 224x224 images.
# We will upsample the 75x75 input images to this resolution.
IMAGE_SIZE = 224
INPUT_CHANNELS = 3  # Band 1, Band 2, Mean(Band 1, Band 2)

# ==========================================
# Model Hyperparameters
# ==========================================
# Dimension of the feature vector extracted from ResNet-18 before the final FC layer
RESNET_FEATURE_DIM = 512
# The incidence angle is a single scalar feature
ANGLE_DIM = 1
# Dropout rate for the classification head
DROPOUT_RATE = 0.5

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4  # L2 regularization
PATIENCE = 5  # Early stopping patience
SEED = 42

# ==========================================
# Hardware Settings
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # Number of subprocesses for data loading
