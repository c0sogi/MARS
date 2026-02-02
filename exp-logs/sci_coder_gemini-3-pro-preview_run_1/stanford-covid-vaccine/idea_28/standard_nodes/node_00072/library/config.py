import os
import torch

# -----------------------------------------------------------------------------
# Directory and File Paths
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_28"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Paths to the pre-generated metadata Parquet files
TRAIN_PARQUET = os.path.join(METADATA_DIR, "train.parquet")
VAL_PARQUET = os.path.join(METADATA_DIR, "val.parquet")
TEST_PARQUET = os.path.join(METADATA_DIR, "test.parquet")

# Output paths
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

# -----------------------------------------------------------------------------
# Data Configuration
# -----------------------------------------------------------------------------
SEQ_LEN = 107
PRED_LEN = 68

# Training targets (subset of available ground truth as per strategy)
TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
NUM_TARGETS = len(TARGET_COLS)

# Vocabulary Mappings
TOKEN2INT = {x: i for i, x in enumerate("ACGU")}
LOOP2INT = {x: i for i, x in enumerate("BEHIMSX")}

NUM_TOKENS = len(TOKEN2INT)
NUM_LOOP_TYPES = len(LOOP2INT)

# -----------------------------------------------------------------------------
# Model Hyperparameters
# -----------------------------------------------------------------------------
# Architecture: Scalar-Aggregated Wide-Stream BiGRU
EMBED_DIM = 32  # Compact embeddings (Cite solution_lesson_node_00049)
HIDDEN_DIM = 384  # Optimal width for RNNs (Cite solution_lesson_node_00070, solution_lesson_node_00022)
NUM_LAYERS = 6  # Shallow and wide backbone
DROPOUT = 0.1  # Standard dropout rate
NOISE_SIGMA = 0.0  # Removed noise to match clean baseline

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = (
    32  # Reduced to maintain gradient update budget (Cite solution_lesson_node_00056)
)
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4  # Low weight decay (relying on noise/dropout for reg)
MAX_GRAD_NORM = 1.0
NUM_WORKERS = 4

# -----------------------------------------------------------------------------
# System Configuration
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
