import os
import torch


class Config:
    """
    Configuration for the Wide-State Identity-Residual Physics-Injected Composite Network.

    This configuration implements the strategy of maintaining a wide, constant latent state
    throughout the backbone with strict identity residuals and physics-informed feature injection.
    """

    # --- Directories ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory specifically for this idea to handle caching safely
    WORKING_DIR = "./working/idea_18"
    SUBMISSION_DIR = "./submission"

    # --- File Paths ---
    # Using metadata files which guarantee breath-id separation
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # --- Reproducibility ---
    SEED = 42

    # --- Model Architecture ---
    # Stem: Multi-scale CNN
    STEM_KERNELS = [3, 5, 7]

    # Backbone: Wide-State Identity-Residual
    D_MODEL = 512
    # Hidden size per direction. 256 * 2 = 512, matching D_MODEL for identity mapping
    LSTM_HIDDEN = 256
    N_BLOCKS = 4
    # Pointwise FFN expansion factor (2x = 1024 hidden)
    FFN_EXPANSION = 2
    DROPOUT = 0.1

    # Deep Supervision
    AUX_WEIGHT = 0.3

    # --- Training Hyperparameters ---
    EPOCHS = 35
    BATCH_SIZE = 256  # A100 40GB can handle large batches
    LR_MAX = 1e-3
    WEIGHT_DECAY = 1e-2
    # Strict gradient clipping is mandatory for Wide-State LSTM stability
    CLIP_GRAD = 1.0
    PCT_START = 0.3  # OneCycleLR warm-up

    # --- Hardware ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # --- Feature Engineering ---
    # Target Variable
    TARGET_COL = "pressure"

    # Identifiers
    ID_COL = "id"
    BREATH_COL = "breath_id"

    # Continuous Features (To be scaled using RobustScaler)
    # Includes raw inputs, physics integrals, interactions, and dynamics
    CONT_FEATURES = [
        "time_step",
        "u_in",
        "R",
        "C",
        "volume",  # Physics: Cumulative integral of flow
        "u_in_R",  # Physics: Interaction term
        "volume_div_C",  # Physics: Interaction term
        "u_in_lag1",  # Dynamics: Lag features
        "u_in_lag2",
        "u_in_lag3",
        "u_in_lag4",
        "u_in_diff1",  # Dynamics: 1st Derivative
        "u_in_diff2",  # Dynamics: 2nd Derivative
    ]

    # Binary Features (NOT scaled, kept as 0/1 for masking)
    BINARY_FEATURES = ["u_out"]

    def __init__(self, debug=False):
        """
        Initialize the configuration.

        Args:
            debug (bool): If True, adjusts hyperparameters for a quick debugging run.
        """
        self.debug = debug

        # Ensure working and submission directories exist
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)

        if self.debug:
            print("!!! DEBUG MODE ENABLED !!!")
            self.EPOCHS = 2
            self.BATCH_SIZE = 64
            # Dataset subsampling should be handled by the data loader using self.debug
