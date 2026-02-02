import os
import torch


class Config:
    """
    Configuration for the Ventilator Pressure Prediction Task.
    Implements the 'ReZero-Stabilized Deeply Supervised Physics-Injected Hybrid Network' strategy.
    """

    # ==========================================
    # Directories and Paths
    # ==========================================
    # Metadata directories (Read-Only inputs)
    METADATA_DIR = "./metadata"
    INPUT_DIR = "./input"

    # Working directory for caching and outputs
    WORKING_DIR = "./working/idea_15"

    # Input Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache Files (Parquet for data, NPY for scaler stats to avoid pickle)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_engineered.parquet")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_engineered.parquet")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_engineered.parquet")

    # Scaler stats paths
    SCALER_CENTER = os.path.join(WORKING_DIR, "scaler_center.npy")
    SCALER_SCALE = os.path.join(WORKING_DIR, "scaler_scale.npy")

    # Model Checkpoint and Submission
    MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    HIDDEN_DIM = 512
    NUM_BLOCKS = 4
    STEM_KERNELS = [3, 5, 7]  # Multi-scale CNN kernel sizes
    EXPANSION_FACTOR = 2  # Pointwise FFN expansion
    DROPOUT = 0.1
    REZERO_INIT = 0.1  # Initial value for ReZero scalars

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    EPOCHS = 35
    BATCH_SIZE = 512
    LR_MAX = 1e-3
    WEIGHT_DECAY = 1e-2
    AUX_WEIGHT = 0.3  # Weight for auxiliary loss (Deep Supervision)
    GRAD_CLIP = 1000.0  # High clip value or None, as ReZero stabilizes

    # Optimizer & Scheduler
    PCT_START = 0.3  # OneCycleLR warm-up percentage
    DIV_FACTOR = 25.0  # OneCycleLR initial divisor
    FINAL_DIV_FACTOR = 1e4  # OneCycleLR final divisor

    # ==========================================
    # Data & Feature Engineering
    # ==========================================
    SEED = 42
    SEQ_LEN = 80  # Fixed sequence length per breath
    ID_COL = "id"
    BREATH_COL = "breath_id"
    TARGET_COL = "pressure"

    # Raw features from dataset
    RAW_FEATURES = ["time_step", "u_in", "u_out", "R", "C"]

    # Engineered features to be generated
    # volume: Cumulative sum of flow * dt
    # u_in_R: Interaction term u_in * R
    # vol_C: Interaction term volume / C
    # u_in_lag1..4: Previous time step values
    # u_in_diff1..2: First and second order differences
    ENG_FEATURES = [
        "volume",
        "u_in_R",
        "vol_C",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_lag3",
        "u_in_lag4",
        "u_in_diff1",
        "u_in_diff2",
    ]

    # Final list of features used for model input
    # Order must be preserved between training and inference
    FEATURE_LIST = RAW_FEATURES + ENG_FEATURES
    INPUT_DIM = len(FEATURE_LIST)

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    def __init__(self, debug=False):
        """
        Initialize configuration.

        Args:
            debug (bool): If True, reduces epochs and dataset size for quick testing.
        """
        # Ensure working directory exists
        os.makedirs(self.WORKING_DIR, exist_ok=True)

        if debug:
            self.EPOCHS = 2
            self.BATCH_SIZE = 64
            self.NUM_WORKERS = 0
            print(
                f"[Config] Debug mode active: Epochs={self.EPOCHS}, Batch={self.BATCH_SIZE}"
            )

    def get_feature_indices(self):
        """Returns a dictionary mapping feature names to their index in the input tensor."""
        return {name: i for i, name in enumerate(self.FEATURE_LIST)}
