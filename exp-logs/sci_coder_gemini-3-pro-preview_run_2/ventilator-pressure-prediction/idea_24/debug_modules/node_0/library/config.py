import os
import torch


class Config:
    """
    Global configuration for the Ventilator Pressure Prediction task.
    Implements the 'Uncompressed Physics-Context Deep-Injection BiLSTM' strategy.
    """

    # ==========================================
    # 1. Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Cache directory for this specific idea/iteration
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_24")

    # Submission directory
    SUBMISSION_DIR = "./submission"

    # Ensure writable directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input File Paths
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata File Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output File Paths
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
    BEST_MODEL_PATH = os.path.join(CACHE_DIR, "best_model.pth")

    # ==========================================
    # 2. Feature Engineering & Data Processing
    # ==========================================
    # Target Variable
    TARGET_COL = "pressure"
    ID_COL = "id"
    BREATH_ID_COL = "breath_id"

    # Continuous Features -> Apply RobustScaler
    # Includes raw physics (R, C), controls (u_in), and engineered physics terms
    CONTINUOUS_FEATURES = [
        "time_step",
        "u_in",
        "R",
        "C",
        "dt",  # Time delta
        "u_in_cumsum",  # Integrated volume (Time-Weighted)
        "R_u_in",  # Resistive pressure interaction
        "u_in_cumsum_div_C",  # Elastic pressure interaction
        "u_in_diff1",  # 1st Finite Difference
        "u_in_diff2",  # 2nd Finite Difference
    ]

    # Binary Features -> Keep Raw (No Scaling)
    # Scaling binary features was identified as a failure mode (Lesson 00062)
    BINARY_FEATURES = ["u_out"]

    # Combined feature dimension for model input
    INPUT_DIM = len(CONTINUOUS_FEATURES) + len(BINARY_FEATURES)

    # ==========================================
    # 3. Model Hyperparameters
    # ==========================================
    # Architecture: Uncompressed Physics-Context Deep-Injection BiLSTM
    HIDDEN_SIZE = 512
    NUM_LAYERS = 4
    DROPOUT = 0.1

    # Wide-Bandwidth Injection Settings
    INJECTION_GLU_WIDTH = 256  # Width of the GLU path (Path B)
    USE_BIDIRECTIONAL = True

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    # Stretched-Horizon Convergence Protocol
    EPOCHS = 200
    BATCH_SIZE = 512  # Optimized for A100 40GB

    # Optimizer
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    COSINE_T_MAX = 200  # Matches EPOCHS for full horizon stretch

    # Loss Function Weights
    INSPIRATORY_WEIGHT = 1.0
    EXPIRATORY_WEIGHT = 0.1  # Reduced weight for expiratory phase

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 25

    # ==========================================
    # 5. Hardware & Reproducibility
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Utilizing available vCPUs
    SEED = 42

    # ==========================================
    # 6. Debugging
    # ==========================================
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 2000  # Number of breaths to use if DEBUG is True
