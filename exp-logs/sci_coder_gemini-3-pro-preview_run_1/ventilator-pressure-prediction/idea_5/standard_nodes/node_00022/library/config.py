import os
import torch

# --- General Configuration ---
SEED = 42
IDEA_NAME = "idea_7"
WORKING_DIR = os.path.join("./working", IDEA_NAME)
CACHE_DIR = WORKING_DIR

# Ensure the working directory exists for caching and outputs
os.makedirs(WORKING_DIR, exist_ok=True)

# --- File Paths ---
# Using metadata splits as requested
TRAIN_CSV = "./metadata/train.csv"
VAL_CSV = "./metadata/val.csv"
TEST_CSV = "./metadata/test.csv"
SAMPLE_SUBMISSION_CSV = "./input/sample_submission.csv"

# Output paths
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "model.pth")
SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

# --- Data Column Definitions ---
ID_COL = "id"
BREATH_ID_COL = "breath_id"
TIME_COL = "time_step"
TARGET_COL = "pressure"

# The dataset consists of 80-step breaths
SEQ_LEN = 80

# --- Feature Engineering Config ---
# Lung attributes to be passed through Learnable Embedding Layers
CATEGORICAL_FEATURES = ["R", "C"]

# Continuous features for the Deep Branch (Multi-Scale CNN + LSTM)
# Includes raw signals and engineered physics proxies
CONTINUOUS_FEATURES = [
    "time_step",
    "u_in",
    "u_out",
    "cumulative_volume",  # Integral of u_in over time
    "flow_interaction",  # u_in * R
    "volume_interaction",  # cumulative_volume / C
]

# Features strictly for the Linear Physics Adapter Branch
# These allow the model to learn a direct linear correction based on the Equation of Motion
PHYSICS_FEATURES = ["flow_interaction", "volume_interaction"]

# --- Training Hyperparameters ---
BATCH_SIZE = 512
EPOCHS = 20  # Fixed budget as per strategy
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
MAX_GRAD_NORM = 1000

# Scheduler (OneCycleLR) settings
MAX_LR = 1e-3
PCT_START = 0.3
ANNEAL_STRATEGY = "cos"
DIV_FACTOR = 25.0
FINAL_DIV_FACTOR = 1000.0

# --- Model Architecture ---
# Embeddings
EMBEDDING_DIM = 8

# Multi-Scale CNN Stem
CNN_KERNELS = [3, 5, 7]
CNN_FILTERS = 64
CNN_DROPOUT = 0.1

# LSTM Backbone
LSTM_HIDDEN_SIZE = 256
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.1
BIDIRECTIONAL = True

# --- Hardware ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
