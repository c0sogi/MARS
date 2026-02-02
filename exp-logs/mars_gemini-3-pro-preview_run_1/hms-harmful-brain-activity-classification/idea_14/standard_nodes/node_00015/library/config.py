import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    PROJECT_NAME = "HarmfulBrainActivityDetection"
    IDEA_NAME = "idea_14"  # Cyclic-Subset Coordinate-Guided Fusion
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging

    # =========================================================================
    # Directory Paths
    # =========================================================================
    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", IDEA_NAME)

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata files (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Data directories
    TRAIN_EEGS_DIR = os.path.join(INPUT_DIR, "train_eegs")
    TRAIN_SPECS_DIR = os.path.join(INPUT_DIR, "train_spectrograms")
    TEST_EEGS_DIR = os.path.join(INPUT_DIR, "test_eegs")
    TEST_SPECS_DIR = os.path.join(INPUT_DIR, "test_spectrograms")

    # Output paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    OUTPUT_SUBMISSION_PATH = "./submission/submission.csv"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)
    # Ensure submission directory exists
    os.makedirs(os.path.dirname(OUTPUT_SUBMISSION_PATH), exist_ok=True)

    # =========================================================================
    # Data Processing Parameters
    # =========================================================================
    # EEG
    EEG_SR = 200  # Original Sampling Rate
    TARGET_SR = 100  # Downsampled Rate
    EEG_DURATION = 50  # Seconds
    EEG_CHANNELS = 20  # Number of EEG channels
    EEG_SEQ_LEN = TARGET_SR * EEG_DURATION  # 5000 time steps

    # Spectrogram
    SPEC_WINDOW = 600  # 10 minutes in seconds
    SPEC_RESIZE_H = 512  # Height (Time)
    SPEC_RESIZE_W = 512  # Width (Frequency)
    SPEC_CHANNELS = 5  # 4 regions + 1 coordinate map

    # Labels
    TARGET_COLS = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    NUM_CLASSES = 6

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Stream A: EEG Encoder
    EEG_KERNELS = [3, 5, 7, 9]
    EEG_FILTERS = [64, 128, 256, 512]

    # Stream B: Spectrogram Encoder
    SPEC_BACKBONE = "tf_efficientnet_b0_ns"
    SPEC_PRETRAINED = True

    # Fusion
    ATTENTION_HEADS = 4
    EMBED_DIM = 512  # Dimension to project both streams to before fusion
    DROPOUT = 0.5

    # =========================================================================
    # Training Strategy (Cyclic-Subset)
    # =========================================================================
    # Cyclic Pipeline Settings
    NUM_FOLDS = 4  # Divide training data into N disjoint folds
    NUM_CYCLES = 2  # How many times to cycle through all folds
    TOTAL_EPOCHS = NUM_FOLDS * NUM_CYCLES  # Total training epochs

    # Optimization
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1.0
    PATIENCE = 3  # Early stopping patience (checks validation every epoch)

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Augmentation
    # =========================================================================
    FREQ_MASK_PARAM = 30
    TIME_MASK_PARAM = 30
    EEG_CHANNEL_DROPOUT_PROB = 0.2
    MODALITY_DROPOUT_PROB = 0.1  # Probability of zeroing out one stream during training

    @classmethod
    def print_config(cls):
        print(f"Configuration: {cls.PROJECT_NAME} ({cls.IDEA_NAME})")
        print(f"Device: {cls.DEVICE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(
            f"Total Epochs: {cls.TOTAL_EPOCHS} ({cls.NUM_CYCLES} cycles of {cls.NUM_FOLDS} folds)"
        )
        print(f"Working Directory: {cls.WORKING_DIR}")
