import os
import torch


class Config:
    """
    Configuration class for the RNA Degradation Prediction project.
    Implements parameters for the Direct-Access Recurrent Dense Network (DA-RDN).
    """

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for Idea 54 (DA-RDN)
    WORKING_DIR = "./working/idea_54"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # File Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Checkpoint Path
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # =========================================================================
    # Data Parameters
    # =========================================================================
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # Columns
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    NUM_TARGETS = len(TARGET_COLS)

    # Feature Flags
    USE_PARTNER_IDENTITY = True  # Explicit Partner Identity Injection

    # Cache Keys (Versioned for DA-RDN)
    CACHE_TRAIN_KEY = "train_data_da_rdn_v1"
    CACHE_VAL_KEY = "val_data_da_rdn_v1"
    CACHE_TEST_KEY = "test_data_da_rdn_v1"

    # =========================================================================
    # Model Architecture Parameters (DA-RDN)
    # =========================================================================
    # Main Backbone (Direct-Access Dense Dilated TCN)
    HIDDEN_DIM = 64  # Growth Rate and Main Channel Dim
    LATENT_DIM = 64  # Projected Latent Dimension Z
    DILATIONS = [1, 2, 4, 8, 16, 32]
    KERNEL_SIZE = 3

    # Pure-Feedback Module
    FEEDBACK_DIM = 32  # Output channels of feedback module
    FEEDBACK_GROWTH_RATE = 16  # Constrained growth rate for feedback
    FEEDBACK_DILATIONS = [1, 2, 4, 8]

    # General
    DROPOUT = 0.1

    # =========================================================================
    # Training Parameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 64
    LR = 1e-3
    EPOCHS = 50
    PATIENCE = 10  # Early stopping patience
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Loss Weights
    LOSS_WEIGHT_FINAL = 1.0
    LOSS_WEIGHT_AUX = 0.5  # For the first pass prediction

    # =========================================================================
    # Debugging / Development
    # =========================================================================
    DEBUG = False  # Set to True to run on a small subset
    MAX_DEBUG_SAMPLES = 100  # Number of samples to use in debug mode

    @classmethod
    def get_cache_path(cls, name):
        """Returns the full path for a cache file in the working directory."""
        return os.path.join(cls.WORKING_DIR, f"{name}.npz")
