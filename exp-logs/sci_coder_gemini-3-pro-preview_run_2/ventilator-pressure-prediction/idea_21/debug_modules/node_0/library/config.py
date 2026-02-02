import os
import torch


class Config:
    """
    Configuration for the Ventilator Pressure Prediction task.
    Implements the 'Fidelity-Preserving Bottlenecked-Context BiLSTM' strategy.
    """

    # ==========================================
    # 1. File Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_21"
    SUBMISSION_DIR = "./submission"

    # Input Files
    TRAIN_DATA_PATH = os.path.join(INPUT_DIR, "train.csv")
    TEST_DATA_PATH = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output/Cache Files
    # Using Parquet for efficient storage of processed tensors/dataframes
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")

    # Scaler statistics (saved as numpy dictionary)
    SCALER_CACHE_PATH = os.path.join(WORKING_DIR, "scaler_params.npy")

    # Model Checkpoint
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Final Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. Data & Feature Configuration
    # ==========================================
    SEED = 42

    # Column Definitions
    ID_COL = "id"
    BREATH_ID_COL = "breath_id"
    TIME_COL = "time_step"
    TARGET_COL = "pressure"

    # Feature Engineering Strategy: Segregated Scaling
    # 1. Continuous Features -> RobustScaler
    # Includes raw physics (u_in, R, C, time) and derived dynamics (integral, deltas)
    CONTINUOUS_FEATURES = [
        "time_step",
        "u_in",
        "R",
        "C",
        "u_in_cumsum",  # Integral
        "u_in_diff1",  # Multi-step Delta t-1
        "u_in_diff2",  # Multi-step Delta t-2
        "u_in_diff3",  # Multi-step Delta t-3
        "u_in_diff4",  # Multi-step Delta t-4
    ]

    # 2. Binary Features -> Identity (No Scaling)
    # Critical correction: Scaling binary flags distorts their logic
    BINARY_FEATURES = ["u_out"]

    # Combined list for convenience in model input dimension calculation
    ALL_FEATURES = CONTINUOUS_FEATURES + BINARY_FEATURES
    INPUT_DIM = len(ALL_FEATURES)

    # ==========================================
    # 3. Model Architecture (FPBC-BiLSTM)
    # ==========================================
    # Wide Deep Recurrent Backbone
    LSTM_HIDDEN_SIZE = 512
    LSTM_LAYERS = 4
    LSTM_DROPOUT = 0.0  # Dropout applied between layers manually in model definition if needed, or via param

    # Monolithic Context Extractor
    GLU_WIDE_SIZE = 256
    CONTEXT_BOTTLENECK_SIZE = 128

    # General
    MODEL_DROPOUT = 0.1
    USE_INPUT_INJECTION = True  # Concatenate raw input to LSTM layers

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    # Stretched-Horizon Convergence Protocol
    EPOCHS = 200

    # Hardware Optimization (A100 40GB)
    BATCH_SIZE = 1024
    NUM_WORKERS = 4

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    CLIP_GRAD_NORM = 1000.0  # High clip for stability with LSTMs

    # Scheduler: Cosine Annealing
    T_MAX = 200  # Matches Epochs
    ETA_MIN = 1e-5

    # Loss Function Weights
    INSPIRATORY_WEIGHT = 1.0
    EXPIRATORY_WEIGHT = 0.1

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Ensures necessary directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on module import
Config.setup()
