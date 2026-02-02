import os

# =========================================================================================
# File Paths & Directories
# =========================================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"

# Cache directory specific to this idea/experiment
CACHE_DIR = os.path.join(WORKING_DIR, "idea_75")
os.makedirs(CACHE_DIR, exist_ok=True)

# Output submission path
SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

# =========================================================================================
# Dataset Configuration
# =========================================================================================
SEQ_LENGTH = 107
SEQ_SCORED = 68

# The full list of target columns present in the training data
TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

# Indices of the columns that are actually scored in the competition metric
# Indices correspond to: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
SCORED_COLS_INDICES = [0, 1, 3]

# =========================================================================================
# Training Hyperparameters
# =========================================================================================
BATCH_SIZE = 16  # Strictly set to 16 as per strategy
LEARNING_RATE = 1e-3  # AdamW learning rate
NUM_EPOCHS = 50  # Maximum epochs
EARLY_STOPPING_PATIENCE = 10
SEED = 42  # Fixed random seed for reproducibility

# =========================================================================================
# Model Hyperparameters (HC-HSGFN)
# =========================================================================================
# Main Backbone (High-Capacity Dense Dilated TCN)
GROWTH_RATE = 64
DILATIONS = [1, 2, 4, 8, 16, 32]
KERNEL_SIZE = 3
DROPOUT = 0.1
LATENT_DIM = 64  # Dimension Z

# Feedback Module (Global-Context Pure-Feedback)
FEEDBACK_GROWTH_RATE = 16  # Lightweight backbone for feedback
FEEDBACK_DIM = 32  # Dimension E_fb

# Interaction & Aggregation
RNN_HIDDEN_SIZE = 64  # Compact hidden size for GRU

# =========================================================================================
# Optimization / Loss Configuration
# =========================================================================================
# Weights for the iterative refinement loss
LOSS_PASS_1_WEIGHT = 0.5
LOSS_PASS_2_WEIGHT = 1.0
