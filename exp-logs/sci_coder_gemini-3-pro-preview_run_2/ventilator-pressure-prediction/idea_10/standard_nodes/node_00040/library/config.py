import os
import torch


class Config:
    # ==========================================
    # 1. Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_10"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Raw Data Paths
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Processed Data Cache Paths (Parquet format preferred over pickle)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_processed.parquet")

    # Model & Output Paths
    MODEL_CHECKPOINT = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. Global Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to train on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 2000  # Number of breaths to use in debug mode
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # 3. Data & Feature Configuration
    # ==========================================
    ID_COL = "id"
    BREATH_ID_COL = "breath_id"
    TARGET_COL = "pressure"
    TIME_COL = "time_step"

    # List of features to be generated and used by the model.
    # This includes raw inputs and engineered physics/dynamics features.
    FEATURE_COLS = [
        "time_step",
        "u_in",
        "u_out",
        "R",
        "C",
        "u_in_cumsum",  # Time-weighted integral of u_in (Volume proxy)
        "R_u_in",  # Interaction: R * u_in (Resistive Pressure)
        "u_in_cumsum_div_C",  # Interaction: Volume / C (Elastic Pressure)
        "u_in_lag1",  # Lag 1
        "u_in_lag2",  # Lag 2
        "u_in_diff1",  # 1st Derivative (Velocity)
        "u_in_diff2",  # 2nd Derivative (Acceleration)
    ]

    # Input dimension for the model
    INPUT_DIM = len(FEATURE_COLS)

    # ==========================================
    # 4. Model Architecture (RGI-BiLSTM)
    # ==========================================
    HIDDEN_DIM = 512
    NUM_LAYERS = 4
    DROPOUT = 0.1  # Applied within recurrent blocks (after LSTM, before next layer)
    USE_GLU = True  # Enable Gated Linear Unit for projection
    USE_INPUT_INJECTION = True  # Enable Input Injection to deep layers

    # ==========================================
    # 5. Training Hyperparameters
    # ==========================================
    EPOCHS = 150
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1000.0

    # Scheduler Settings (Cosine Annealing)
    SCHEDULER_T_MAX = 150
    SCHEDULER_MIN_LR = 1e-6

    # Loss Weights
    INSPIRATORY_WEIGHT = 1.0
    EXPIRATORY_WEIGHT = 0.1
