import os
import torch


class Config:
    """
    Configuration for the Deeply-Supervised Coordinate-Fusion Network.
    Implements parameters for Idea 15:
    - Dual-Stream (EEG + Spectrogram)
    - Coordinate Injection (5th channel for spectrograms)
    - Global Random Subsampling (20k samples, 5 epochs)
    - Multi-Task KL Divergence Loss
    """

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_15"
    SUBMISSION_DIR = "./submission"

    # Raw Data Directories
    TRAIN_EEGS_DIR = os.path.join(INPUT_DIR, "train_eegs")
    TRAIN_SPECS_DIR = os.path.join(INPUT_DIR, "train_spectrograms")
    TEST_EEGS_DIR = os.path.join(INPUT_DIR, "test_eegs")
    TEST_SPECS_DIR = os.path.join(INPUT_DIR, "test_spectrograms")

    # Metadata Files (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Paths
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Processing Parameters
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # EEG Signal Processing
    EEG_CHANNELS = 20  # 19 EEG leads + 1 EKG
    EEG_RAW_SAMPLING_RATE = 200  # Hz
    EEG_TARGET_SAMPLING_RATE = 100  # Hz (Downsampling)
    EEG_DURATION = 50  # seconds
    # Sequence length = 50s * 100Hz = 5000 samples
    EEG_SEQ_LEN = int(EEG_DURATION * EEG_TARGET_SAMPLING_RATE)

    # Spectrogram Processing
    # 4 standard regions (LL, RL, LP, RP) + 1 Coordinate Map
    SPEC_CHANNELS = 5
    SPEC_RESIZE_SIZE = (512, 512)  # (Height, Width)

    # Labels
    # Training targets (probabilities)
    TARGET_COLS = [
        "seizure_prob",
        "lpd_prob",
        "gpd_prob",
        "lrda_prob",
        "grda_prob",
        "other_prob",
    ]
    # Submission columns (votes)
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
    # Training Strategy (Global Random Subsampling)
    # =========================================================================
    # Instead of full dataset, sample 20k and train for multiple epochs
    # to allow BN statistics to stabilize and model to converge.
    TRAIN_SAMPLE_SIZE = 20000

    EPOCHS = 5
    BATCH_SIZE = 32

    # Optimization
    LEARNING_RATE = 1e-3
    MAX_LR = 1e-3  # For OneCycleLR
    WEIGHT_DECAY = 1e-2

    # Multi-Task Loss Weights
    # L_total = L_joint + 0.5*L_eeg + 0.5*L_spec
    LOSS_WEIGHT_JOINT = 1.0
    LOSS_WEIGHT_EEG = 0.5
    LOSS_WEIGHT_SPEC = 0.5

    # =========================================================================
    # Model Architecture
    # =========================================================================
    MODEL_NAME = "DeeplySupervisedCoordFusion"
    BACKBONE_NAME = "efficientnet_b0"  # Lightweight backbone for Spectrograms

    # =========================================================================
    # Hardware
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Initialize the working environment by creating necessary directories.
        """
        dirs = [cls.WORKING_DIR, cls.CACHE_DIR, cls.CHECKPOINT_DIR, cls.SUBMISSION_DIR]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        print(f"Configuration setup complete. Working directory: {cls.WORKING_DIR}")
