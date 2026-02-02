import os
import torch


class Config:
    # ==============================
    # General Settings
    # ==============================
    PROJECT_NAME = "Anchored_Hybrid_Dense_RN"
    IDEA_NAME = "idea_81"
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2  # Number of dataloader workers

    # ==============================
    # File Paths
    # ==============================
    # Input Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directory (Cache & Outputs)
    WORKING_DIR = os.path.join("./working", IDEA_NAME)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Cache Files (Explicit Cache Invalidation as requested)
    CACHE_TRAIN = os.path.join(WORKING_DIR, "train_data_ahd_rn_v1.npz")
    CACHE_VAL = os.path.join(WORKING_DIR, "val_data_ahd_rn_v1.npz")
    CACHE_TEST = os.path.join(WORKING_DIR, "test_data_ahd_rn_v1.npz")

    # Model Checkpoint
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join("./submission", "submission.csv")

    # Ensure submission dir exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # ==============================
    # Data Configuration
    # ==============================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Target Columns (Ground Truth)
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Columns used for scoring in the competition metric
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    NUM_TARGETS = len(TARGET_COLS)  # 5

    # ==============================
    # Model Architecture
    # ==============================
    # Input Features
    # 4 bases (A,G,C,U) + 1 partner base identity + 3 structure (.,(,)) + 7 loop types
    # Note: Partner base identity is 4 channels (one-hot).
    # Total input channels depends on specific encoding implementation, usually around 18-20.

    # Backbone (Post-Activation Dense Dilated TCN)
    GROWTH_RATE = 48  # Capacity setting
    LATENT_DIM = 64  # Projection dimension Z
    KERNEL_SIZE = 3
    DILATIONS = [1, 2, 4, 8, 16, 32]
    DROPOUT = 0.1

    # Global-Context Pure-Feedback Module
    FB_GROWTH_RATE = 16  # Lightweight Dense TCN for feedback
    FB_LATENT_DIM = 32  # Feedback embedding dimension E_fb

    # Aggregation
    RNN_HIDDEN_SIZE = 64  # Compact hidden size
    RNN_LAYERS = 1
    RNN_BIDIRECTIONAL = True

    # ==============================
    # Training Hyperparameters
    # ==============================
    # Optimization Strategy: Small Batch Regime
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Training Loop
    NUM_EPOCHS = 50  # Max epochs
    PATIENCE = 10  # Early stopping patience

    # Loss Weights
    # L_total = MCRMSE(Y_2) + AUX_WEIGHT * MCRMSE(Y_1)
    AUX_WEIGHT = 0.5
