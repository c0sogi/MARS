import os
import torch

# ------------------------------------------------------------------------------
# Directories
# ------------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_23"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# File Paths
# ------------------------------------------------------------------------------
# Raw Data
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Outputs
PROCESSED_DATA_PATH = os.path.join(WORKING_DIR, "processed_data.npz")
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ------------------------------------------------------------------------------
# Data Configuration
# ------------------------------------------------------------------------------
# Number of continuous features (f_00 to f_30, excluding f_27)
NUM_CONTINUOUS_FEATURES = 30

# Sequence configuration for f_27
SEQUENCE_LENGTH = 10
# Vocab size: 26 letters (A-Z). We use 1-based indexing (1-26), 0 for padding.
VOCAB_SIZE = 27

# ------------------------------------------------------------------------------
# Model Architecture Configuration
# ------------------------------------------------------------------------------
# Transformer Stream (Stream 1)
EMBED_DIM = 32
TRANSFORMER_LAYERS = 2
TRANSFORMER_HEADS = 4
TRANSFORMER_DROPOUT = 0.1
TRANSFORMER_ACTIVATION = "gelu"

# Backbone & Fusion (Stream 2)
# The hidden dimension after fusing stream 1 and 2, before the backbone
FUSION_PROJECTION_DIM = 512

# Sustained-Depth Backbone structure
BACKBONE_DIMS = [512, 256, 128]
BLOCKS_PER_STAGE = 3
BACKBONE_DROPOUT = 0.35

# ------------------------------------------------------------------------------
# Training Configuration
# ------------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 1024
EPOCHS = 40

# Optimization
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2

# Scheduler (StepLR)
LR_STEP_SIZE = 10
LR_GAMMA = 0.1

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
