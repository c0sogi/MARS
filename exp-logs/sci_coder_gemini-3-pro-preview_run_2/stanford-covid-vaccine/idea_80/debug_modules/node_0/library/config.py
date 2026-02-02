import os
import torch


class Config:
    """
    Central configuration for the RNA Degradation Prediction task.
    Implements parameters for the Anchored Spatial-Dense Recurrent Network (AS-DRN).
    """

    # --------------------------------------------------------------------------
    # Directory & File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific cache directory for this idea iteration
    CACHE_DIR = "./working/idea_80"
    SUBMISSION_DIR = "./submission"

    # Raw Input Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Stratified Splits)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Dimensions & Targets
    # --------------------------------------------------------------------------
    SEQ_LENGTH = 107
    SCORED_LENGTH = 68

    # All 5 targets provided in training data
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # The 3 targets actually scored for the competition metric
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    NUM_TARGETS = len(TARGET_COLS)

    # --------------------------------------------------------------------------
    # Model Hyperparameters (AS-DRN)
    # --------------------------------------------------------------------------
    # Input Processing
    USE_PARTNER_IDENTITY = True

    # Main Backbone (High-Capacity Dense Dilated TCN)
    KERNEL_SIZE = 3
    DILATIONS = [1, 2, 4, 8, 16, 32]
    MAIN_GROWTH_RATE = 64
    DROPOUT = 0.1

    # Latent Projection & Feedback Module
    LATENT_DIM = 64
    FEEDBACK_DIM = 32
    FB_GROWTH_RATE = 16

    # Aggregation (Bidirectional GRU)
    HIDDEN_DIM = 64

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    # Optimization
    BATCH_SIZE = 16  # Strictly set to 16 as per Lesson 00129
    LR = 1e-3
    EPOCHS = 50
    PATIENCE = 10  # For Early Stopping

    # System
    NUM_WORKERS = 2
    SEED = 42

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------------------------
    # Runtime / Debugging
    # --------------------------------------------------------------------------
    DEBUG = False
    DEBUG_SAMPLES = 200  # Subset size for debugging

    @staticmethod
    def setup_directories():
        """Creates necessary working directories if they don't exist."""
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# Initialize directories immediately upon module import
Config.setup_directories()
