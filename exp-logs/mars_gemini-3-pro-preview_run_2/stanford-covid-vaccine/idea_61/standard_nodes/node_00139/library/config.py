import os
import torch


class Config:
    # ==============================
    # File Paths & Directories
    # ==============================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_61"

    # Input Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Cache Files (Explicit Cache Invalidation with new keys)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data_ads_rn_v1.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data_ads_rn_v1.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data_ads_rn_v1.npz")

    # Output Submission
    SUBMISSION_PATH = "./submission/submission.csv"
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # ==============================
    # Data Dimensions
    # ==============================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Feature Dimensions
    # Sequence (4) + Structure (3) + LoopType (7) = 14
    # Partner Identity (4) = 4
    # Total Input Channels = 18
    INPUT_CHANNELS = 18

    # ==============================
    # Model Architecture (ADS-RN)
    # ==============================
    # Main Backbone (Post-Activation Dense Dilated TCN)
    HIDDEN_DIM = 64
    GROWTH_RATE = 64
    DILATIONS = [1, 2, 4, 8, 16, 32]
    KERNEL_SIZE = 3
    DROPOUT = 0.1
    LATENT_DIM = 64

    # Spatial-Stem Feedback Module
    FEEDBACK_INPUT_CHANNELS = (
        5  # reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    )
    FEEDBACK_GROWTH_RATE = 16
    FEEDBACK_OUT_DIM = 32

    # ==============================
    # Training Hyperparameters
    # ==============================
    BATCH_SIZE = 16  # Strictly set to 16 as per Lesson 00129
    LR = 1e-3
    EPOCHS = 50  # Sufficient for convergence with early stopping
    PATIENCE = 10  # Early stopping patience
    MAX_GRAD_NORM = 5.0

    # Loss Weights
    AUX_LOSS_WEIGHT = 0.5  # Weight for the first pass prediction

    # ==============================
    # System & Reproducibility
    # ==============================
    SEED = 42
    NUM_WORKERS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Ensures the working directory exists."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        # Ensure submission directory exists as well
        os.makedirs(os.path.dirname(cls.SUBMISSION_PATH), exist_ok=True)
