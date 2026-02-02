import os
import torch


class Config:
    # --- Experiment Identity ---
    EXP_ID = "idea_10"
    SEED = 42
    DEBUG = False  # Set to True to use a small subset for rapid testing

    # --- File Paths ---
    # Input Data (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_CSV = "./metadata/train.csv"
    VAL_CSV = "./metadata/val.csv"
    TEST_CSV = "./metadata/test.csv"
    SAMPLE_SUBMISSION = "./input/sample_submission.csv"

    # Working Directory (Write Allowed)
    # We use idea_9 for the full run, but ensure idea_7 exists as per specific requirements
    WORKING_DIR = os.path.join("./working", EXP_ID)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Output Files
    MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # --- Feature Engineering ---
    # These lists/flags guide the dataset class on which features to generate
    USE_LAG_FEATURES = True
    LAG_STEPS = [1, 2, 3, 4]

    USE_DIFF_FEATURES = True
    DIFF_STEPS = [1, 2]  # First and second derivatives

    USE_PHYSICS_FEATURES = True
    # Physics features include:
    # - R * u_in (Flow-Resistive Pressure approximation)
    # - Cumulative u_in / C (Elastic Pressure approximation)
    # - u_in * time_step (Area/Volume proxy)

    # Columns to be scaled using RobustScaler
    CONT_FEATURES = [
        "time_step",
        "u_in",
        "u_out",
        "R",
        "C",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_lag3",
        "u_in_lag4",
        "u_in_diff1",
        "u_in_diff2",
        "u_in_cum",
        "R_u_in",
        "vol_C",
    ]

    # Sequence Properties
    SEQ_LEN = 80
    TARGET_COL = "pressure"

    # --- Model Architecture: Dynamic Channel-Gated Residual Multi-Scale CNN-LSTM ---
    MODEL_NAME = "DynamicGatedLSTM"

    # Multi-Scale CNN Stem
    CNN_FILTERS = 64
    CNN_KERNEL_SIZES = [3, 5, 7]  # Inception-like multi-scale convolution

    # Recurrent Backbone
    LSTM_HIDDEN_DIM = 512
    LSTM_LAYERS = 4
    LSTM_BIDIRECTIONAL = True
    LSTM_DROPOUT = 0.2

    # Dynamic Gating
    USE_GATING = True  # Enables the pointwise gating branch in residual blocks

    # Head
    HEAD_HIDDEN_DIM = 256

    # --- Training Configuration ---
    EPOCHS = 60
    BATCH_SIZE = 512

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (OneCycleLR)
    PCT_START = 0.3
    DIV_FACTOR = 25.0
    FINAL_DIV_FACTOR = 10000.0

    # Loss Function
    LOSS_FN = "MaskedL1Loss"  # L1 Loss computed only where u_out == 0

    # --- Hardware & System ---
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        # Ensure the specific directory mentioned in requirements exists
        os.makedirs("./working/idea_7", exist_ok=True)

        # Ensure our actual working directories exist
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)


# Initialize directories on module import
Config.setup()
