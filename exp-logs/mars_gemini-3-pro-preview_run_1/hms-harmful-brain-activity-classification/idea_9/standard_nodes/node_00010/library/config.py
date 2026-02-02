import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    PROJECT_NAME = "HMS_Brain_Activity_Classification"
    IDEA_NAME = "idea_9"  # Time-Relative Transformer Decoder Network

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of dataloader workers

    # =========================================================================
    # File Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Data Directories
    TRAIN_EEGS_DIR = os.path.join(INPUT_DIR, "train_eegs")
    TRAIN_SPECS_DIR = os.path.join(INPUT_DIR, "train_spectrograms")
    TEST_EEGS_DIR = os.path.join(INPUT_DIR, "test_eegs")
    TEST_SPECS_DIR = os.path.join(INPUT_DIR, "test_spectrograms")

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Directories
    OUTPUT_DIR = os.path.join("./working", IDEA_NAME)
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Ensure output directories exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # =========================================================================
    # Data Configuration
    # =========================================================================
    # EEG Settings
    EEG_RAW_SAMPLE_RATE = 200  # Hz
    EEG_TARGET_SAMPLE_RATE = 100  # Hz (Downsampling as per idea)
    EEG_DURATION = 50  # Seconds
    EEG_SEQ_LEN = EEG_DURATION * EEG_TARGET_SAMPLE_RATE  # 5000 time steps
    EEG_CHANNELS = 20  # 19 EEG + 1 EKG

    # Spectrogram Settings
    SPEC_DURATION = 600  # 10 minutes (600 seconds)
    SPEC_SIZE = (
        512,
        512,
    )  # Resize resolution (Freq, Time) or (Time, Freq) depending on usage

    # Classes
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
    # Model Architecture Configuration
    # =========================================================================
    # Stream A: Raw EEG Encoder (Multi-Scale 1D CNN)
    EEG_KERNELS = [3, 5, 7, 9]  # Kernel sizes for Inception-like blocks
    EEG_CNN_CHANNELS = 32  # Base channel dimension for CNN

    # Stream B: Global Spectrogram Encoder
    # Using EfficientNet-B2 as backbone
    SPEC_BACKBONE = "tf_efficientnet_b2.ns_jft_in1k"
    SPEC_EMBED_DIM = 256  # Dimension to project backbone features to

    # Fusion: Transformer Decoder
    D_MODEL = 256  # Transformer hidden dimension
    NHEAD = 8  # Number of attention heads
    NUM_DECODER_LAYERS = 2  # Number of decoder layers
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1

    # Relative Temporal Embedding
    # Max offset in seconds (approx 10 mins = 600s, half is 300s)
    MAX_REL_TIME_OFFSET = 400

    # =========================================================================
    # Training Configuration
    # =========================================================================
    BATCH_SIZE = 32
    EPOCHS = 6

    # Optimizer
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.01
    MAX_LR = 1e-3  # For OneCycleLR

    # Scheduler
    PCT_START = 0.3  # Percentage of training to increase LR
    DIV_FACTOR = 25
    FINAL_DIV_FACTOR = 1000

    # Loss
    # We use KL Divergence, which expects log-probabilities

    # Debugging / Development
    DEBUG = False  # Set to True to run on a small subset of data
    DEBUG_SAMPLE_SIZE = 1000
