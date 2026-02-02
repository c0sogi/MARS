import os

# =============================================================================
# Directories and File Paths
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_23"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data Paths (using metadata splits)
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "validation.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Cache Paths for processed data
TRAIN_CACHE = os.path.join(WORKING_DIR, "train_processed_v3.parquet")
VAL_CACHE = os.path.join(WORKING_DIR, "val_processed_v3.parquet")
TEST_CACHE = os.path.join(WORKING_DIR, "test_processed_v3.parquet")
SCALER_CACHE = os.path.join(WORKING_DIR, "scaler_v3.joblib")

# Model and Submission Output Paths
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# Global Hyperparameters
# =============================================================================
SEED = 42
BATCH_SIZE = 128  # Strictly 128 to avoid sharp minima and BN instability
EPOCHS = 80  # Extended training for hybrid architecture convergence
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
CLIP_GRAD_NORM = 1.0  # Mandatory gradient clipping
DEBUG = False  # Set to True to restrict dataset size for debugging

# =============================================================================
# Model Architecture Hyperparameters (PE-RDH-Net)
# =============================================================================
# Branch 2: High-Capacity Bidirectional LSTM (Elastic Stream)
LSTM_HIDDEN_SIZE = 512
LSTM_LAYERS = 3
LSTM_BIDIRECTIONAL = True

# Branch 1: Deep Residual Dense TCN (Resistive Stream)
# Increased capacity to match LSTM branch (Cite solution_lesson_node_00057)
CNN_KERNEL_SIZE = 9
CNN_FILTERS = 256
CNN_DROPOUT = 0.1

# Fusion Head: Wide-Latent Integration
WIDE_HIDDEN_SIZE = 1024
FINAL_DROPOUT = 0.1

# =============================================================================
# Feature Engineering Configuration
# =============================================================================
ID_COL = "id"
BREATH_ID_COL = "breath_id"
TIME_COL = "time_step"
TARGET_COL = "pressure"

# 1. Raw Input Features
RAW_FEATURES = ["u_in", "u_out", "R", "C"]

# 2. Derived Physical Features
# dt: Time delta (time_step_t - time_step_{t-1})
# area: Numerical integration of u_in (Volume)
# du_in: Acceleration (u_in_t - u_in_{t-1})
# R_u_in: Interaction term R * u_in
# area_C: Interaction term Area / C
PHYSICAL_FEATURES = ["dt", "area", "du_in", "R_u_in", "area_C"]

# 3. Lookahead Features
# Explicitly shifted columns for u_in(t+1 ... t+4)
LOOKAHEAD_STEPS = 4
LOOKAHEAD_FEATURES = [f"u_in_next_{i}" for i in range(1, LOOKAHEAD_STEPS + 1)]

# 4. Positional Encoding Features
# REMOVED: Explicit positional encodings degrade performance in sequential models
# (Cite solution_lesson_node_00085)

# Final Feature List
# This list defines the exact input vector structure for the model
FEATURE_COLS = RAW_FEATURES + PHYSICAL_FEATURES + LOOKAHEAD_FEATURES

# Input Dimension derived from feature list
INPUT_DIM = len(FEATURE_COLS)
