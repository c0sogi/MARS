import os
import torch


class Config:
    """
    Configuration for the Dual-Stream Physics-Informed Network (DSPIN) pipeline.
    Centralizes file paths, hyperparameters, and feature engineering settings.
    """

    # ====================================================
    # General Configuration
    # ====================================================
    SEED = 42
    DEBUG = False  # Set to True to use a small subset of data for debugging
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = os.cpu_count() or 4

    # ====================================================
    # Paths
    # ====================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"

    # Raw Data Files
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Pre-generated splits)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    # Cache files will be saved here (e.g., train_cache.parquet)
    CACHE_DIR = WORKING_DIR
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ====================================================
    # Data & Feature Engineering
    # ====================================================
    # Base columns to load
    # Note: R and C are continuous physical attributes, u_out is binary control
    BASE_COLS = ["time_step", "u_in", "u_out", "R", "C"]
    TARGET_COL = "pressure"
    ID_COL = "id"
    BREATH_ID_COL = "breath_id"

    # Physics-Based Feature Flags
    USE_LAGS = True
    LAG_STEPS = [1, 2, 3, 4]  # Previous time steps for temporal context

    USE_DIFFS = True
    DIFF_STEPS = [1, 2]  # 1st and 2nd derivatives (Velocity, Acceleration)

    USE_CUMSUM = True  # Integral of flow (Volume approximation)

    USE_INTERACTIONS = True  # Physics terms: R*Flow, Volume/C

    # Normalization
    SCALER_TYPE = "robust"  # RobustScaler to handle outliers in sensor data

    # ====================================================
    # Model Architecture: Dual-Stream Physics-Informed Network
    # ====================================================
    # Common
    HIDDEN_DIM = 256

    # Stream 1: Recurrent Path (Models Elastic Pressure / Volume Dynamics)
    # Deep Bidirectional LSTM with Input Injection
    LSTM_LAYERS = 4
    LSTM_BIDIRECTIONAL = True
    LSTM_DROPOUT = 0.1

    # Stream 2: Instantaneous Path (Models Resistive Pressure / Flow Dynamics)
    # Deep Residual MLP
    MLP_LAYERS = 4
    MLP_DIM = 256
    MLP_DROPOUT = 0.1

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    BATCH_SIZE = 512  # Large batch size for A100 efficiency
    EPOCHS = 60  # Sufficient for convergence without CNN overhead

    # Optimizer
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1000.0

    # Loss Function Weights
    # We weight the inspiratory phase (u_out=0) higher as it is the scored metric.
    # Expiratory phase (u_out=1) is kept with low weight for state stability.
    LOSS_INSPIRATORY_WEIGHT = 1.0
    LOSS_EXPIRATORY_WEIGHT = 0.1

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 15

    @classmethod
    def setup(cls):
        """Creates the working directory if it does not exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
