import os
import torch


class Config:
    # ==========================================
    # General Configuration
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    EXP_NAME = "idea_22"

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", EXP_NAME)

    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data & Feature Engineering
    # ==========================================
    # Raw columns provided in the dataset
    ID_COL = "id"
    BREATH_ID_COL = "breath_id"
    TIME_COL = "time_step"
    TARGET_COL = "pressure"

    # Physics & Interaction Features Configuration
    # These names must match the keys generated in the feature engineering pipeline

    # Continuous features to be scaled using RobustScaler
    CONTINUOUS_FEATURES = [
        "time_step",
        "u_in",
        "R",
        "C",
        "u_in_cumsum",  # Integral of u_in * dt (Volume proxy)
        "R_u_in",  # R * u_in (Resistive pressure component)
        "u_in_cumsum_div_C",  # u_in_cumsum / C (Elastic pressure component)
        "u_in_diff1",  # Multi-step delta t-1
        "u_in_diff2",  # Multi-step delta t-2
        "u_in_diff3",  # Multi-step delta t-3
        "u_in_diff4",  # Multi-step delta t-4
    ]

    # Binary features to be kept Raw (No scaling)
    BINARY_FEATURES = ["u_out"]

    # Total input dimension for the model
    INPUT_DIM = len(CONTINUOUS_FEATURES) + len(BINARY_FEATURES)

    # ==========================================
    # Model Architecture (PADI-BiLSTM)
    # ==========================================
    MODEL_NAME = "PADI_BiLSTM"

    # Backbone
    LSTM_HIDDEN_SIZE = 512
    LSTM_NUM_LAYERS = 4
    LSTM_DROPOUT = 0.2  # Dropout between LSTM layers

    # Injection Block
    # The injection payload is (Identity + Bottlenecked GLU)
    # Identity dim is INPUT_DIM.
    # GLU outputs a bottleneck representation.
    INJECTION_BOTTLENECK_DIM = 128

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    EPOCHS = 200
    BATCH_SIZE = 512  # Optimized for A100

    # Optimizer
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = 200  # Matches EPOCHS for stretched horizon
    ETA_MIN = 1e-6

    # Loss Weights
    # Weighted L1: Inspiratory phase (u_out=0) gets weight 1.0
    # Expiratory phase (u_out=1) gets weight 0.1
    LOSS_WEIGHT_INSPIRATORY = 1.0
    LOSS_WEIGHT_EXPIRATORY = 0.1

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """Ensures necessary directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Run setup on import to ensure directories exist
Config.setup()
