import os
import torch
import numpy as np
import random


class Config:
    # ==============================
    # 1. File Paths & Directories
    # ==============================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_9"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Files (Pre-split)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Artifacts
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
    MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")

    # Cache Directory (for .npy / .parquet files)
    CACHE_DIR = WORKING_DIR

    # ==============================
    # 2. Data Configuration
    # ==============================
    # Column Definitions
    ID_COL = "id"
    BREATH_ID_COL = "breath_id"
    TIME_COL = "time_step"
    TARGET_COL = "pressure"
    U_IN_COL = "u_in"
    U_OUT_COL = "u_out"
    R_COL = "R"
    C_COL = "C"

    # Sequence Parameters
    N_STEPS = 80  # Fixed time steps per breath

    # Feature Engineering Configuration
    # The pipeline will generate these features if they don't exist
    FEATURE_COLS = [
        "time_step",
        "u_in",
        "u_out",
        "R",
        "C",
        # Dynamics (Inertia)
        "u_in_lag1",
        "u_in_lag2",
        "u_in_lag3",
        "u_in_lag4",
        "u_in_diff1",
        "u_in_diff2",
        # Physics-Fidelity Terms
        "volume",  # Integral of u_in * dt
        "R_u_in",  # Interaction: R * u_in (Pressure drop due to resistance)
        "vol_C",  # Interaction: volume / C (Pressure due to elastance)
    ]

    # Deep Context Injection Features
    # Subset of features to be concatenated to the input of *every* LSTM layer
    # to preserve physical constraints throughout the network depth.
    CONTEXT_FEATURES = ["R", "C", "R_u_in", "vol_C"]

    # ==============================
    # 3. Model Architecture
    # ==============================
    # Stem: Multi-Scale CNN
    CNN_KERNELS = [3, 5, 7]
    CNN_FILTERS = 64

    # Backbone: Residual Bi-LSTM
    LSTM_HIDDEN_DIM = 256
    LSTM_LAYERS = 4
    BIDIRECTIONAL = True

    # Regularization
    DROPOUT = 0.1

    # ==============================
    # 4. Training Hyperparameters
    # ==============================
    SEED = 42
    EPOCHS = 35
    BATCH_SIZE = 512

    # Optimization (AdamW + OneCycleLR)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler Params
    PCT_START = 0.3
    DIV_FACTOR = 25.0
    FINAL_DIV_FACTOR = 1000.0

    # Loss Function
    MASK_EXPIRATORY = True  # Only train on inspiratory phase (u_out == 0)

    # ==============================
    # 5. Runtime & Debugging
    # ==============================
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 2000  # Number of breaths to use when DEBUG=True

    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def set_seed(seed=42):
        """Sets the random seed for full reproducibility."""
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @classmethod
    def get_data_hash_config(cls):
        """
        Returns a dictionary of configuration parameters that affect data processing.
        This is used to generate a hash for caching processed datasets.
        """
        return {
            "feature_cols": cls.FEATURE_COLS,
            "n_steps": cls.N_STEPS,
            "debug": cls.DEBUG,
            "debug_size": cls.DEBUG_SAMPLE_SIZE,
            "seed": cls.SEED,
        }
