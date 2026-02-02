import os
import torch

# Ensure working directory and submission directory exist
os.makedirs("./working/idea_26/", exist_ok=True)
os.makedirs("./submission/", exist_ok=True)


class Config:
    """
    Configuration module for the Direct-Kinematic Residual-Hybrid Network (DKRH-Net).
    Defines global hyperparameters, file paths, and feature engineering specifications.
    """

    # ==========================================
    # File Paths & Directories
    # ==========================================
    # Input metadata paths (Pre-split by breath_id)
    TRAIN_PATH = "./metadata/train.csv"
    VAL_PATH = "./metadata/validation.csv"
    TEST_PATH = "./metadata/test.csv"

    # Reference sample submission
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Working directory for caching processed data (parquet/npy) and checkpoints
    WORKING_DIR = "./working/idea_26/"

    # Final submission output path
    SUBMISSION_PATH = "./submission/submission.csv"

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Strictly enforced Batch Size of 128 (Balances gradient noise and BN stability)
    BATCH_SIZE = 128

    # High budget epochs to allow synchronization of distinct gradient dynamics
    EPOCHS = 80

    # Standard learning rate
    LEARNING_RATE = 1e-3

    # Low Weight Decay (1e-4) to avoid underfitting unscaled regression targets
    WEIGHT_DECAY = 1e-4

    # Gradient Clipping to prevent explosion in LSTM/Deep CNN
    MAX_GRAD_NORM = 1.0

    # Early Stopping Patience
    PATIENCE = 15

    # Compute Settings
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Model Architecture (DKRH-Net)
    # ==========================================
    # Branch 1: Deep Residual Dense TCN (Resistive Stream)
    # Uses large kernels and dense convolutions for derivative modeling
    CNN_FILTERS = 64
    CNN_KERNEL_SIZE = 9
    CNN_LAYERS = 6  # Number of Residual Dense Blocks
    CNN_DROPOUT = 0.1

    # Branch 2: High-Capacity Bi-LSTM (Elastic Stream)
    # Serves as the numerical integrator
    LSTM_HIDDEN_SIZE = 512
    LSTM_LAYERS = 3
    LSTM_BIDIRECTIONAL = True
    LSTM_DROPOUT = 0.1

    # Fusion Head: Wide-Latent Integration
    DENSE_HIDDEN_SIZE = 1024

    # ==========================================
    # Feature Engineering
    # ==========================================
    # Complete Kinematic State Engineering
    # These columns must be generated during preprocessing and present in the input tensor
    FEATURE_COLS = [
        "u_in",  # Proportional control
        "u_out",  # Expiratory valve status (retained for context)
        "R",  # Lung Resistance
        "C",  # Lung Compliance
        "u_in_diff1",  # Backward Velocity (Momentum): u_in(t) - u_in(t-1)
        "u_in_lead1",  # Forward Lookahead (Intent): t+1
        "u_in_lead2",  # Forward Lookahead (Intent): t+2
        "u_in_lead3",  # Forward Lookahead (Intent): t+3
        "u_in_lead4",  # Forward Lookahead (Intent): t+4
        "dt",  # Time Delta: time_step(t) - time_step(t-1)
        "area",  # Volume Integration: sum(u_in * dt)
        "R_u_in",  # Physical Interaction: R * u_in
        "area_C",  # Physical Interaction: Area / C
    ]

    # Input dimension derived from feature columns
    INPUT_DIM = len(FEATURE_COLS)

    # Target Variable
    TARGET_COL = "pressure"

    # Columns to exclude from X (features) if present in the DataFrame
    # raw 'time_step' is excluded as it destabilizes CNNs; 'id'/'breath_id' are metadata
    DROP_COLS = ["id", "breath_id", "time_step", "pressure"]
