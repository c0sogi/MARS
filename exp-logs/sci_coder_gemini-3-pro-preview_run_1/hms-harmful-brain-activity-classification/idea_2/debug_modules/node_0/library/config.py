import os
import torch


class Config:
    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific output directory for Idea 2 (Multi-Scale 1D CNN)
    OUTPUT_DIR = os.path.join(WORKING_DIR, "idea_2")
    SUBMISSION_DIR = "./submission"

    # Data Directories
    TRAIN_EEGS_DIR = os.path.join(INPUT_DIR, "train_eegs")
    TEST_EEGS_DIR = os.path.join(INPUT_DIR, "test_eegs")
    TRAIN_SPECS_DIR = os.path.join(INPUT_DIR, "train_spectrograms")
    TEST_SPECS_DIR = os.path.join(INPUT_DIR, "test_spectrograms")

    # Metadata Files (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Submission Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Checkpoint
    MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
    PREDS_PATH = os.path.join(OUTPUT_DIR, "predictions.npy")

    # =========================================================================
    # Data Parameters
    # =========================================================================
    SAMPLING_RATE = 200  # Original Hz
    TARGET_RATE = 100  # Downsampled Hz
    DURATION = 50  # Seconds
    FIXED_LENGTH = int(DURATION * TARGET_RATE)  # 5000 samples

    # EEG Channels (Standard 10-20 system + EKG)
    # Order matches the parquet file columns
    EEG_CHANNELS = [
        "Fp1",
        "F3",
        "C3",
        "P3",
        "F7",
        "T3",
        "T5",
        "O1",
        "Fz",
        "Cz",
        "Pz",
        "Fp2",
        "F4",
        "C4",
        "P4",
        "F8",
        "T4",
        "T6",
        "O2",
        "EKG",
    ]
    NUM_CHANNELS = 20

    # Target Columns (Probabilities from metadata)
    TARGET_COLS = [
        "seizure_prob",
        "lpd_prob",
        "gpd_prob",
        "lrda_prob",
        "grda_prob",
        "other_prob",
    ]
    # Output Columns for Submission
    OUTPUT_COLS = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    NUM_CLASSES = 6

    # =========================================================================
    # Model Hyperparameters (Multi-Scale 1D CNN)
    # =========================================================================
    # Inception-style kernels for capturing different frequency bands
    KERNELS = [3, 5, 7, 9, 11]

    # Channel dimensions for the convolutional blocks
    # Starts with input channels, then expands
    HIDDEN_DIMS = [64, 128, 256]

    DROPOUT = 0.5
    USE_RESIDUAL = True

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 64  # Adjusted for A100 GPU (40GB)
    EPOCHS = 15  # Sufficient for convergence with OneCycle

    # Optimizer & Scheduler
    LEARNING_RATE = 1e-3  # Base LR
    MAX_LR = 1e-2  # Max LR for OneCycle
    WEIGHT_DECAY = 1e-2  # For AdamW
    PCT_START = 0.3  # Percentage of training to increase LR

    # Early Stopping
    PATIENCE = 4
    MIN_DELTA = 1e-4

    # =========================================================================
    # Compute & Debugging
    # =========================================================================
    NUM_WORKERS = 4  # Number of dataloader workers (12 vCPUs available)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging flags
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SUBSET_SIZE = 2000  # Number of samples to use if DEBUG is True

    @classmethod
    def setup(cls):
        """
        Creates necessary output and working directories.
        """
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set reproducible seeds
        import random
        import numpy as np

        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
