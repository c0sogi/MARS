import os
import torch


class Config:
    """
    Configuration for the Pyramid-Resolution Coordinate-Guided Fusion Network.
    Centralizes all file paths, data shapes, model hyperparameters, and training settings.
    """

    # =========================================================================
    # 1. File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_16"

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

    # Output Directories
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # 2. Data Configuration
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 4

    # EEG Stream Parameters
    EEG_RAW_SAMPLE_RATE = 200
    EEG_TARGET_SAMPLE_RATE = 100  # Downsample to 100 Hz
    EEG_DURATION_SEC = 50
    EEG_SEQ_LEN = EEG_DURATION_SEC * EEG_TARGET_SAMPLE_RATE  # 5000 time steps
    EEG_CHANNELS = 20  # 19 EEG electrodes + 1 EKG

    # Spectrogram Stream Parameters
    SPEC_DURATION_SEC = 600  # 10 minutes
    SPEC_SIZE = (512, 512)  # (Frequency, Time) or (Height, Width)
    # 5 Channels: LL, RL, LP, RP, + Coordinate Map
    SPEC_CHANNELS = 5

    # Target Labels
    TARGET_COLS = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    PROB_COLS = [
        "seizure_prob",
        "lpd_prob",
        "gpd_prob",
        "lrda_prob",
        "grda_prob",
        "other_prob",
    ]
    NUM_CLASSES = 6

    # =========================================================================
    # 3. Model Architecture Configuration
    # =========================================================================
    # Stream B: Spectrogram Encoder (EfficientNet + FPN)
    BACKBONE = "tf_efficientnet_b0_ns"
    PRETRAINED = True
    FPN_OUT_CHANNELS = 128  # Dimension of features after FPN fusion

    # Stream A: Raw EEG Encoder (Multi-Scale 1D CNN)
    EEG_KERNELS = [3, 5, 7, 9]  # Inception-style kernel sizes
    EEG_FILTERS = [32, 32, 32, 32]  # Filters per parallel branch
    EEG_EMBED_DIM = 128  # Output dimension of EEG stream (must match FPN_OUT_CHANNELS)

    # Fusion: Asymmetric Cross-Attention
    ATTN_NUM_HEADS = 4
    ATTN_DROPOUT = 0.1

    # =========================================================================
    # 4. Training Configuration
    # =========================================================================
    # Global Random Subsampling Strategy
    TRAIN_SUBSAMPLE_SIZE = 20000  # Number of samples to train on per run

    BATCH_SIZE = 32
    EPOCHS = 5
    LR = 1e-3
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1.0
    PATIENCE = 3  # Early stopping patience

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Ensures all necessary working and output directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Directories setup complete at {cls.WORKING_DIR}")
