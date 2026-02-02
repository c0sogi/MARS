import os
import torch


class Config:
    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    OUTPUT_DIR = "./working/idea_13"
    SUBMISSION_DIR = "./submission"

    # Input Data Folders
    TRAIN_EEGS_DIR = os.path.join(INPUT_DIR, "train_eegs")
    TRAIN_SPECS_DIR = os.path.join(INPUT_DIR, "train_spectrograms")
    TEST_EEGS_DIR = os.path.join(INPUT_DIR, "test_eegs")
    TEST_SPECS_DIR = os.path.join(INPUT_DIR, "test_spectrograms")

    # Metadata Files (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Processing Constants
    # =========================================================================
    # EEG
    EEG_ORIGINAL_SR = 200
    EEG_TARGET_SR = 100  # Downsample to 100Hz
    EEG_DURATION = 50  # Seconds
    EEG_SEQ_LEN = EEG_DURATION * EEG_TARGET_SR  # 5000 samples
    EEG_CHANNELS = 20

    # Spectrogram
    SPEC_DURATION = 600  # 10 minutes
    SPEC_SIZE = (512, 512)  # (Height/Time, Width/Freq)
    SPEC_CHANNELS = 5  # 4 regions (LL, RL, LP, RP) + 1 Coordinate Map

    # Labels
    CLASS_NAMES = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
    NUM_CLASSES = len(CLASS_NAMES)
    TARGET_COLS = [f"{c}_prob" for c in CLASS_NAMES]  # Training targets
    OUTPUT_COLS = [f"{c}_vote" for c in CLASS_NAMES]  # Submission columns

    # =========================================================================
    # Model Architecture
    # =========================================================================
    MODEL_NAME = "AuxiliarySupervisedFusionNet"
    BACKBONE_SPEC = "efficientnet_b0"
    BACKBONE_EEG = "inception_1d"
    PRETRAINED = True

    # Regularization
    DROPOUT_RATE = 0.2
    MODALITY_DROPOUT_PROB = 0.2  # Probability to drop an entire stream during training

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 2  # Limited epochs for full dataset training

    # Optimizer & Scheduler
    LR = 1e-3
    WEIGHT_DECAY = 1e-2
    MAX_LR = 1e-3  # For OneCycleLR
    PCT_START = 0.3  # For OneCycleLR

    # Loss
    AUX_LOSS_WEIGHT = 0.5  # Lambda for auxiliary heads

    # Compute
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories.
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
