import os
import torch


class Config:
    # =========================================================================
    # System & Reproducibility
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_40"
    SUBMISSION_DIR = "./submission"

    # Metadata Files (Parquet)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.parquet")

    # Model Artifacts & Caching
    # Cache directory for processed .npz files
    CACHE_DIR = WORKING_DIR
    # Path to save the best model checkpoint
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    # Path for the final submission file
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Input Channels:
    # 4 (Nucleotides: A, G, C, U)
    # 3 (Structure: (, ), .)
    # 7 (Loop Type: S, M, I, B, H, E, X)
    INPUT_CHANNELS = 14

    # Output Targets: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    OUTPUT_CHANNELS = 5

    # Indices of columns used for the competition metric (MCRMSE)
    # 0: reactivity, 1: deg_Mg_pH10, 3: deg_Mg_50C
    SCORED_INDICES = [0, 1, 3]

    # =========================================================================
    # Model Hyperparameters (Deep Decoupled Channel-Gated BiGRU)
    # =========================================================================
    HIDDEN_DIM = 384  # High capacity backbone
    NUM_LAYERS = 4  # Deep architecture
    KERNEL_SIZE = 3  # For Convolutional Stem
    DROPOUT = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 64
    LR = 1e-3
    EPOCHS = 50

    # Optimization & Stability
    GRADIENT_CLIP = 1.0  # Mandatory for stability
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10  # Early stopping patience

    # Scheduler settings (CosineAnnealingLR)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # =========================================================================
    # Debugging & Flexibility
    # =========================================================================
    # Set to True to run on a small subset for testing pipeline integrity
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100
    DEBUG_EPOCHS = 2

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_run_params(cls, debug=False):
        """
        Returns a dictionary of parameters, adjusting for debug mode if requested.
        """
        params = {
            "epochs": cls.DEBUG_EPOCHS if debug else cls.EPOCHS,
            "subset_size": cls.DEBUG_SUBSET_SIZE if debug else None,
            "batch_size": cls.BATCH_SIZE,
            "num_workers": cls.NUM_WORKERS,
        }
        return params
