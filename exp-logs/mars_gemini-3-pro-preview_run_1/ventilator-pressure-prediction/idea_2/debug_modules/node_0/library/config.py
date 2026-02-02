import os
import torch


class Config:
    """
    Configuration for the Ventilator Pressure Prediction task.
    Implements settings for the Physics-Enhanced Hybrid CNN-LSTM strategy.
    """

    # ==========================================
    # File Paths & Directories
    # ==========================================
    # Input Data (using generated metadata)
    TRAIN_PATH = "./metadata/train.csv"
    VAL_PATH = "./metadata/val.csv"
    TEST_PATH = "./metadata/test.csv"
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Output Directories
    # Using 'idea_2' to isolate this strategy's artifacts
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # Artifact Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # Data Processing & Feature Engineering
    # ==========================================
    # Sequence length is fixed for this dataset (approx 3s breath)
    SEQ_LEN = 80

    # Column Names
    COL_ID = "id"
    COL_BREATH_ID = "breath_id"
    COL_TIME = "time_step"
    COL_PRESSURE = "pressure"
    COL_U_IN = "u_in"
    COL_U_OUT = "u_out"
    COL_R = "R"
    COL_C = "C"

    # Feature Lists
    # Continuous features to be scaled (RobustScaler recommended)
    CONT_FEATURES = [
        "time_step",
        "u_in",
        "u_in_cumsum",  # Physics: Cumulative volume (integral of flow)
        "u_in_diff1",  # Dynamics: Velocity
        "u_in_diff2",  # Dynamics: Acceleration
        "u_in_lag1",  # Temporal context
        "u_in_lag2",  # Temporal context
        "u_in_R",  # Interaction: Flow * Resistance
        "u_in_C",  # Interaction: Flow * Compliance
    ]

    # Categorical features
    # R and C are discrete physical settings. u_out is binary.
    # We will one-hot encode R and C. u_out is kept as binary feature.
    CAT_FEATURES = ["R", "C"]

    # Unique values for One-Hot Encoding
    R_VALUES = [5, 20, 50]
    C_VALUES = [10, 20, 50]

    # Calculated Input Dimension for the Model
    # Continuous Features (9) + One-Hot R (3) + One-Hot C (3) + u_out (1)
    INPUT_DIM = len(CONT_FEATURES) + len(R_VALUES) + len(C_VALUES) + 1

    # ==========================================
    # Model Architecture (Hybrid CNN-LSTM)
    # ==========================================
    # 1D CNN Encoder
    CNN_FILTERS = 64
    CNN_KERNEL_SIZE = 3

    # Bidirectional LSTM
    LSTM_HIDDEN_SIZE = 512
    LSTM_LAYERS = 4
    LSTM_DROPOUT = 0.1
    BIDIRECTIONAL = True

    # Regression Head
    FC_HIDDEN_SIZE = 128

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Debugging: Set True to train on a small subset of breaths
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 2000  # Number of breaths for debugging

    # Optimization
    EPOCHS = 200
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (OneCycleLR)
    PCT_START = 0.2

    # Loss Function
    # We only compute loss on the inspiratory phase (u_out == 0)
    MASK_EXPIRATORY_PHASE = True

    # Early Stopping
    PATIENCE = 15

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def get_cache_path(cls, filename):
        """Helper to get full path for cached files in working dir."""
        return os.path.join(cls.WORKING_DIR, filename)
