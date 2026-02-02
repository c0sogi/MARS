import os
import torch


class Config:
    """
    Configuration for the Corrected Wide-Context Dense-Hybrid Network (CWDH-Net).
    Defines hyperparameters, file paths, and feature sets for the pipeline.
    """

    # =========================================================================
    # 1. File Paths & Directories
    # =========================================================================
    # Input Metadata (Pre-split)
    TRAIN_PATH = "./metadata/train.csv"
    VAL_PATH = "./metadata/validation.csv"
    TEST_PATH = "./metadata/test.csv"
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Working Directory (Cache & Models)
    WORKING_DIR = "./working/idea_20"
    OUTPUT_DIR = "./submission"

    # Artifact Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SCALER_PATH = os.path.join(WORKING_DIR, "scaler.joblib")
    SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # 2. Data Configuration
    # =========================================================================
    # General
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for testing

    # Feature Engineering Parameters
    N_LAGS = 4  # Number of lookahead steps for u_in

    # Feature Selection
    # Note: 'u_out' is included here as part of the dynamic state input vector.
    # 'time_step' is explicitly excluded to prevent translation variance issues.
    CONT_FEATURES = [
        "u_in",  # Control input
        "u_out",  # Expiratory valve (Crucial for backward pass context)
        "R",  # Resistance
        "C",  # Compliance
        "dt",  # Time delta (variable rate physics)
        "u_in_diff",  # Acceleration (derivative)
        "area",  # Volume (integral)
        "R_uin",  # Interaction: R * u_in
        "area_C",  # Interaction: Area / C
        "u_in_lead1",  # Lookahead t+1
        "u_in_lead2",  # Lookahead t+2
        "u_in_lead3",  # Lookahead t+3
        "u_in_lead4",  # Lookahead t+4
    ]

    # Features to strictly ignore during training
    EXCLUDED_FEATURES = ["id", "breath_id", "pressure", "time_step"]

    # Target Column
    TARGET_COL = "pressure"

    # =========================================================================
    # 3. Model Architecture (CWDH-Net)
    # =========================================================================
    # Common
    INPUT_DIM = len(CONT_FEATURES)

    # Branch 1: Dense Large-Kernel TCN (Resistive Stream)
    TCN_KERNEL_SIZE = 9
    TCN_FILTERS = [64, 128, 256, 512]  # Increasing channel capacity
    TCN_DROPOUT = 0.1

    # Branch 2: Bidirectional LSTM (Elastic Stream)
    LSTM_HIDDEN_DIM = 512
    LSTM_LAYERS = 3
    LSTM_DROPOUT = 0.1

    # Fusion Head
    FUSION_DIM = 1024  # Wide latent integration layer

    # =========================================================================
    # 4. Training Hyperparameters
    # =========================================================================
    # Optimization
    BATCH_SIZE = (
        64  # Reduced for better generalization (Cite solution_lesson_node_00018)
    )
    EPOCHS = 100  # Extended for convergence (Cite solution_lesson_node_00039)
    LEARNING_RATE = 1e-3  # Standard starting rate
    WEIGHT_DECAY = 1e-4

    # Scheduler
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 15

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 40)
        print(f"CWDH-Net Configuration")
        print("=" * 40)
        print(f"Device:        {cls.DEVICE}")
        print(f"Batch Size:    {cls.BATCH_SIZE}")
        print(f"Epochs:        {cls.EPOCHS}")
        print(f"Input Dim:     {cls.INPUT_DIM}")
        print(f"Features:      {cls.CONT_FEATURES}")
        print(f"Working Dir:   {cls.WORKING_DIR}")
        print("=" * 40)
