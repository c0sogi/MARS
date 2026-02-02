import os
import torch

# --- General Configuration ---
SEED = 42
IDEA_NAME = "idea_9"
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
# Lung attributes are treated as continuous inputs (Cite solution_lesson_node_00022)
CATEGORICAL_FEATURES = []

# Continuous features including physics proxies and temporal derivatives (Cite solution_lesson_node_00022)
CONTINUOUS_FEATURES = [
    "time_step",
    "u_in",
    "u_out",
    "R",
    "C",
    "cumulative_volume",
    "flow_interaction",
    "volume_interaction",
    "u_in_lag1",
    "u_in_lag2",
    "u_in_lag3",
    "u_in_lag4",
    "u_in_diff1",
    "u_in_diff2",
    "u_in_diff3",
    "u_in_diff4",
    "u_out_lag1",
    "u_out_lag2",
]

# Removed separate Physics Branch features (Cite solution_lesson_node_00011)
PHYSICS_FEATURES = []

# --- Training Hyperparameters ---
BATCH_SIZE = 512
EPOCHS = (
    15  # Increased epochs to match batch size scaling (Cite solution_lesson_node_00019)
)
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
LSTM_HIDDEN_SIZE = 512  # Increased capacity (Cite solution_lesson_node_00023)
LSTM_LAYERS = 4  # Increased depth (Cite solution_lesson_node_00023)
LSTM_DROPOUT = 0.1
BIDIRECTIONAL = True

# --- Hardware ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
