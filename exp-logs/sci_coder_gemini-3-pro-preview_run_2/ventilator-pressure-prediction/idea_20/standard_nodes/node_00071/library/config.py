import os
import torch


class Config:
    # ==========================================
    # 1. File System & Paths
    # ==========================================
    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_20"
    SUBMISSION_DIR = "./submission"

    # Ensure working/submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Raw Data Files
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Pre-generated)
    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache Files (Parquet/NPY for speed)
    CACHE_TRAIN_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
    CACHE_VAL_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
    CACHE_TEST_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")
    CACHE_SCALER_PATH = os.path.join(WORKING_DIR, "scaler_params.npz")

    # Model Artifacts
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. Reproducibility & Debugging
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLE_SIZE = 1000  # Number of breaths to use in debug mode

    # ==========================================
    # 3. Model Architecture (CWCDP-BiLSTM)
    # ==========================================
    # Wide Deep Recurrent Backbone
    LSTM_HIDDEN_SIZE = 512
    LSTM_LAYERS = 4
    BIDIRECTIONAL = True

    # Wide Monolithic Context Extractor
    GLU_SIZE = 256

    # Regularization
    DROPOUT = 0.1  # Inter-layer dropout

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    # Stretched-Horizon Convergence Protocol
    EPOCHS = 200
    BATCH_SIZE = 512  # A100 allows large batches

    # Optimizer (AdamW)
    LR = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = 200  # Matches EPOCHS
    ETA_MIN = 1e-5

    # Loss Weights
    # Weighted L1: Inspiratory=1.0, Expiratory=0.1
    LOSS_INSPIRATORY_WEIGHT = 1.0
    LOSS_EXPIRATORY_WEIGHT = 0.1

    # ==========================================
    # 5. Feature Engineering & Scaling
    # ==========================================
    # Target Variable
    TARGET_COL = "pressure"

    # Feature Groups for Segregated Scaling
    # Continuous: Apply RobustScaler (Median/IQR)
    CONTINUOUS_FEATURES = [
        "time_step",
        "u_in",
        "R",
        "C",
        # Derived Physics Terms
        "area",  # Time-weighted integral of u_in
        "R_u_in",  # R * u_in (Resistive pressure proxy)
        "area_div_C",  # area / C (Elastic pressure proxy)
        # Multi-Step Deltas (t - (t-k))
        "u_in_diff1",
        "u_in_diff2",
        # Removed diff3, diff4 (Cite 00066)
    ]

    # Binary: Do NOT Scale (Pass Raw)
    BINARY_FEATURES = ["u_out"]

    # Combined list for convenience
    ALL_FEATURES = CONTINUOUS_FEATURES + BINARY_FEATURES

    # Input Dimension calculation
    INPUT_DIM = len(ALL_FEATURES)

    # Hardware settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For DataLoader
