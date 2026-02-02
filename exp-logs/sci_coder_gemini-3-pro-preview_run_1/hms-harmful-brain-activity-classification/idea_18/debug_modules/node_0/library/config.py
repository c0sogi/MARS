import os
import torch


class Config:
    """
    Configuration class for the Bottleneck-Projected Coordinate-Fusion Network.
    Centralizes all hyperparameters for data processing, model architecture, and training
    as per the requirements of Idea 18.
    """

    # =========================================================================
    # 1. Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Output directory for checkpoints, cache, and submissions
    OUTPUT_DIR = os.path.join(WORKING_DIR, "idea_18")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    # =========================================================================
    # 2. Data Configuration
    # =========================================================================
    SEED = 42
    NUM_CLASSES = 6
    CLASS_NAMES = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

    # EEG Parameters (Stream A)
    EEG_SR = 100  # Target sampling rate (downsampled from 200Hz)
    EEG_DURATION = 50  # Duration in seconds
    EEG_SEQ_LEN = 5000  # 50s * 100Hz
    EEG_CHANNELS = 20  # Number of EEG channels

    # Spectrogram Parameters (Stream B)
    SPEC_SIZE = (512, 512)  # Resize dimensions (Height, Width)
    SPEC_CHANNELS = 5  # 4 regions (LL, RL, LP, RP) + 1 coordinate map

    # =========================================================================
    # 3. Model Architecture
    # =========================================================================
    # Stream A: Raw EEG Encoder (Multi-Scale 1D Conv)
    EEG_KERNEL_SIZES = [3, 5, 7, 9]

    # Stream B: Coordinate-Aware Spectrogram Encoder
    ENCODER_NAME = "tf_efficientnet_b0_ns"
    PRETRAINED = True

    # Bottleneck & Fusion
    BOTTLENECK_DIM = 128  # Projection dimension for fusion (Low-Rank Bottleneck)
    ATTENTION_DIM = 128  # Dimension for Cross-Attention
    DROP_RATE = 0.2  # Dropout rate

    # =========================================================================
    # 4. Training Strategy
    # =========================================================================
    # Global Random Subsampling Strategy
    USE_SUBSET = True
    SUBSET_SIZE = 25000  # Fixed subset size per run

    EPOCHS = 4  # Number of epochs
    BATCH_SIZE = 32  # Batch size

    # Optimizer (AdamW) & Scheduler (OneCycle)
    LR = 1e-3
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 10.0

    # System & Compute
    NUM_WORKERS = 4  # Number of data loading workers
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # 5. Caching
    # =========================================================================
    # Directory to store cached processed arrays
    CACHE_DIR = OUTPUT_DIR
