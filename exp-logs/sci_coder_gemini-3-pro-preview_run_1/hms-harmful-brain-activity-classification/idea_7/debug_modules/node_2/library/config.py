import os
import torch
import random
import numpy as np


class Config:
    # =========================================================================
    # 1. Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    # Input Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Data Directories
    TRAIN_EEGS_DIR = os.path.join(INPUT_DIR, "train_eegs")
    TRAIN_SPECS_DIR = os.path.join(INPUT_DIR, "train_spectrograms")
    TEST_EEGS_DIR = os.path.join(INPUT_DIR, "test_eegs")
    TEST_SPECS_DIR = os.path.join(INPUT_DIR, "test_spectrograms")

    # Output Directories (created in setup)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # =========================================================================
    # 2. Data Parameters
    # =========================================================================
    # EEG
    EEG_RAW_SAMPLE_RATE = 200
    EEG_TARGET_SAMPLE_RATE = 100  # Downsample to 100Hz
    EEG_DURATION_SEC = 50
    EEG_SEQ_LEN = EEG_DURATION_SEC * EEG_TARGET_SAMPLE_RATE  # 5000 time steps
    EEG_CHANNELS = 20  # 19 EEG + 1 EKG

    # Spectrogram
    SPEC_DURATION_SEC = 600  # 10 minutes
    SPEC_IMG_SIZE = (512, 512)  # (Height/Time, Width/Freq) for EfficientNet

    # Offset Guidance
    # Sigma for the Gaussian Bias Mask (in terms of relative time [0,1])
    # A smaller sigma focuses attention more tightly around the offset.
    ATTENTION_MASK_SIGMA = 0.1

    # Labels
    CLASS_NAMES = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    NUM_CLASSES = 6

    # =========================================================================
    # 3. Model Hyperparameters
    # =========================================================================
    MODEL_NAME = "OffsetGuidedDualStream"

    # Stream A: EEG
    EEG_BACKBONE = "inception_1d"
    EEG_KERNELS = [3, 5, 7, 9]
    EEG_FILTERS = [32, 64, 128, 256]

    # Stream B: Spectrogram
    SPEC_BACKBONE = "tf_efficientnet_b2_ns"
    SPEC_PRETRAINED = True

    # Fusion
    ATTENTION_DIM = 256
    DROPOUT_RATE = 0.5

    # =========================================================================
    # 4. Training Settings
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization
    EPOCHS = 8
    BATCH_SIZE = 32  # Adjust for A100 40GB
    NUM_WORKERS = 4

    # Learning Rate (OneCycleLR)
    MAX_LR = 1e-3
    PCT_START = 0.2
    DIV_FACTOR = 25.0
    FINAL_DIV_FACTOR = 1000.0
    WEIGHT_DECAY = 1e-2

    # Loss
    USE_KL_DIV = True

    # Debugging / Development
    DEBUG = False
    TRAIN_SUBSET_SIZE = 2000  # Only used if DEBUG=True
    VAL_SUBSET_SIZE = 500  # Only used if DEBUG=True

    @classmethod
    def setup(cls, debug=False):
        """
        Initializes directories and sets random seeds.
        """
        cls.DEBUG = debug

        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set Reproducibility Seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"Config Setup Complete. Device: {cls.DEVICE}, Debug: {cls.DEBUG}")
        if cls.DEBUG:
            print(
                f"Running in DEBUG mode with subset sizes: Train={cls.TRAIN_SUBSET_SIZE}, Val={cls.VAL_SUBSET_SIZE}"
            )
