import os
import torch


class Config:
    """
    Configuration for the Robust Hybrid-Stem Global-Feedback Network (RHS-GFN).
    Defines file paths, data dimensions, model hyperparameters, and training settings.
    """

    # ------------------------------------------------------------------------
    # General & Reproducibility
    # ------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    # ------------------------------------------------------------------------
    # File Paths
    # ------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_69"

    # Input Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache Files (Parquet/NPZ)
    # Explicit versioning to ensure cache safety as per requirements
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data_rhs_gfn_v1.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data_rhs_gfn_v1.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data_rhs_gfn_v1.npz")

    # Output Files
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ------------------------------------------------------------------------
    # Data Configuration
    # ------------------------------------------------------------------------
    SEQ_LEN = 107
    SCORED_LEN = 68

    # Target Definitions
    # Full list of targets present in training data
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Targets used for the competition metric
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # Indices mapping for masking logic
    # 0: reactivity (Scored)
    # 1: deg_Mg_pH10 (Scored)
    # 2: deg_pH10 (Unscored)
    # 3: deg_Mg_50C (Scored)
    # 4: deg_50C (Unscored)
    SCORED_INDICES = [0, 1, 3]
    UNSCORED_INDICES = [2, 4]
    NUM_TARGETS = 5

    # ------------------------------------------------------------------------
    # Model Hyperparameters (RHS-GFN)
    # ------------------------------------------------------------------------
    # Main Backbone: Post-Activation Dense Dilated TCN
    BACKBONE_GROWTH_RATE = 64
    BACKBONE_KERNEL_SIZE = 3
    BACKBONE_DILATIONS = [1, 2, 4, 8, 16, 32]
    BACKBONE_DROPOUT = 0.1
    LATENT_DIM = 64  # Z dimension

    # Global-Context Pure-Feedback Module
    FEEDBACK_GROWTH_RATE = 16
    FEEDBACK_DIM = 32  # E_fb dimension

    # Interaction & Aggregation (RNN)
    RNN_HIDDEN_DIM = 64
    RNN_LAYERS = 1
    RNN_BIDIRECTIONAL = True

    # ------------------------------------------------------------------------
    # Training Hyperparameters
    # ------------------------------------------------------------------------
    BATCH_SIZE = 16  # Strictly set to 16 for gradient frequency
    LEARNING_RATE = 1e-3  # AdamW default
    EPOCHS = 50  # Max epochs
    PATIENCE = 10  # Early stopping patience

    @classmethod
    def setup(cls):
        """Ensures the working directory exists."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)


# Initialize environment on import
Config.setup()
