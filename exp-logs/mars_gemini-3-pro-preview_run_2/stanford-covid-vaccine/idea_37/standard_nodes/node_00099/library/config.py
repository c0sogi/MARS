import os

# =============================================================================
# File Paths & Directories
# =============================================================================
# Base Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"

# Idea-Specific Directory (Idea 37: Structure-Conditional Recurrent Dense Network)
IDEA_ID = "idea_37"
PREPROCESSED_DIR = os.path.join(WORKING_DIR, IDEA_ID)

# Ensure working directory exists
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

# Raw Data Files (Reference)
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")
SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata Files (Stratified Splits)
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Cache Files (Numpy Archives)
TRAIN_CACHE = os.path.join(PREPROCESSED_DIR, "train_data_scr_dn_v1.npz")
VAL_CACHE = os.path.join(PREPROCESSED_DIR, "val_data_scr_dn_v1.npz")
TEST_CACHE = os.path.join(PREPROCESSED_DIR, "test_data_scr_dn_v1.npz")

# Output Files
MODEL_SAVE_PATH = os.path.join(PREPROCESSED_DIR, "best_model.pth")
SUBMISSION_PATH = os.path.join(PREPROCESSED_DIR, "submission.csv")

# =============================================================================
# Data Configuration
# =============================================================================
SEQ_LENGTH = 107
SCORING_LENGTH = 68

# Target Columns
# All 5 targets provided in training
TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
# Only these 3 are used for the competition metric
SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

# =============================================================================
# Model Hyperparameters
# =============================================================================
# Input Embedding Dimension
# 4 (Sequence: A,G,C,U)
# + 3 (Structure: (,.,))
# + 7 (Loop Type: S,M,I,B,H,E,X)
# + 4 (Partner Identity: A,G,C,U or 0 if unpaired)
EMBED_DIM = 18

# Backbone: Static Dense Dilated TCN
HIDDEN_DIM = 64
LAYERS = 6
KERNEL_SIZE = 3
DILATIONS = [1, 2, 4, 8, 16, 32]
DROPOUT = 0.1

# Latent Projection & Conditioning
LATENT_DIM = 64
COND_DIM = 16

# Feedback Module
FEEDBACK_LAYERS = 3
FEEDBACK_CHANNELS = 32

# =============================================================================
# Training Configuration
# =============================================================================
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 25
NUM_PASSES = (
    2  # Number of recycling passes (1st pass zero feedback, 2nd pass with feedback)
)

# Optimization
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
PATIENCE = 5  # Early stopping patience

# Hardware / System
NUM_WORKERS = 2
SEED = 42

# Debugging
# Set DEBUG to True to run on a small subset of data for testing pipeline
DEBUG = False
DEBUG_SUBSET_SIZE = 50
