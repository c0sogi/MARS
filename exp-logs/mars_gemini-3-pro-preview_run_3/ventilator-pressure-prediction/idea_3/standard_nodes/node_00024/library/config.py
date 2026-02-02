import os
import torch


class Config:
    """
    Central configuration for the Ventilator Pressure Prediction pipeline.
    Implements the settings for the Parallel TCN-LSTM Hybrid strategy.
    """

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    # Input Metadata (Read-Only)
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory (Write Access)
    # Used for caching processed data and saving models
    WORKING_DIR = "./working/idea_3"
    CACHE_DIR = WORKING_DIR

    # Output Files
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "model.pth")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Feature Configuration
    # -------------------------------------------------------------------------
    # Defines the exact features to be generated and used by the model.
    # This ensures pipeline integrity between data processing and model input.
    FEATURE_COLS = [
        # Raw Control Inputs & Time
        "time_step",
        "u_in",
        "u_out",
        # Lung Attributes
        "R",
        "C",
        # Engineered Features (PID State & Physics)
        "u_in_cumsum",  # Integral term (Volume proxy)
        "u_in_diff1",  # Derivative term (Flow acceleration)
        "u_in_diff2",  # Second derivative
        "R_u_in",  # Interaction: Resistive pressure component (R * Flow)
        "vol_C_ratio",  # Interaction: Elastic pressure component (Volume / C)
    ]

    # -------------------------------------------------------------------------
    # Model & Training Hyperparameters
    # -------------------------------------------------------------------------
    HYPERPARAMS = {
        # Input Dimension
        "input_dim": len(FEATURE_COLS),
        # Architecture: Parallel TCN-LSTM Hybrid
        # Branch 1: TCN (Fast Dynamics)
        "tcn_channels": 64,
        "tcn_levels": 4,
        "tcn_kernel_size": 3,
        "tcn_dropout": 0.1,
        # Branch 2: LSTM (Slow Integration)
        "lstm_hidden_dim": 128,
        "lstm_layers": 3,
        "lstm_bidirectional": True,
        "lstm_dropout": 0.1,
        # Fusion Head
        "fc_hidden_dim": 128,
        # Optimization
        "learning_rate": 1e-3,
        "batch_size": 128,  # Smaller batch size for better generalization (Cite solution_lesson_node_00018)
        "epochs": 60,
        "patience": 15,  # Early stopping patience
        # Scheduler (ReduceLROnPlateau)
        "scheduler_factor": 0.5,
        "scheduler_patience": 3,
        "min_lr": 1e-6,
        # Regularization
        "weight_decay": 1e-4,
        # Hardware / System
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_workers": 4,
        "pin_memory": True,
    }

    # -------------------------------------------------------------------------
    # Pipeline Control
    # -------------------------------------------------------------------------
    # If True, deletes existing cache to force re-computation of features.
    # Essential for ensuring new feature engineering logic is applied.
    FORCE_RECOMPUTE = True

    # Debugging flag to run on a small subset of data
    DEBUG = False
    DEBUG_BREATHS = 1000
