import os
import torch


class Config:
    """
    Configuration for the Bidirectional Coordinate-Guided Fusion Network.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_11"

    # Raw Data Directories
    TRAIN_EEGS_DIR = os.path.join(INPUT_DIR, "train_eegs")
    TRAIN_SPECTROGRAMS_DIR = os.path.join(INPUT_DIR, "train_spectrograms")
    TEST_EEGS_DIR = os.path.join(INPUT_DIR, "test_eegs")
    TEST_SPECTROGRAMS_DIR = os.path.join(INPUT_DIR, "test_spectrograms")

    # Metadata Files (Generated in ./metadata)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    SUBMISSION_CSV = "./submission/submission.csv"
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    # Caching
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    LOAD_CACHED_DATA = True  # If True, tries to load pre-processed tensors from disk

    # =========================================================================
    # Data Processing Parameters
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 4  # Optimized for the available 12 vCPUs

    # EEG Stream Parameters
    EEG_DURATION = 50  # Seconds
    EEG_ORIGINAL_RATE = 200  # Hz
    EEG_TARGET_RATE = 100  # Hz (Downsampled as per strategy)
    EEG_SEQ_LEN = EEG_DURATION * EEG_TARGET_RATE  # 5000 time steps
    EEG_CHANNELS = 20  # 19 EEG + 1 EKG

    # Spectrogram Stream Parameters
    SPEC_DURATION = 600  # 10 minutes (600 seconds)
    SPEC_IMG_SIZE = (512, 512)  # (Height, Width)
    SPEC_CHANNELS = 5  # 4 Regions (LL, RL, LP, RP) + 1 Coordinate Map

    # Augmentation
    MASK_TIME_PROB = 0.2
    MASK_FREQ_PROB = 0.2
    CHANNEL_DROPOUT_PROB = 0.2

    # =========================================================================
    # Model Architecture Parameters
    # =========================================================================
    # Stream A: Raw EEG Encoder (Multi-Scale 1D CNN)
    EEG_KERNELS = [3, 5, 7, 9]  # Kernel sizes for parallel branches
    EEG_BASE_FILTERS = 32

    # Stream B: Spectrogram Encoder
    SPEC_BACKBONE = "efficientnet_b1"
    SPEC_PRETRAINED = True

    # Fusion: Bidirectional Cross-Attention
    FUSION_HIDDEN_DIM = 256
    ATTENTION_HEADS = 4
    DROPOUT_RATE = 0.2

    # Output
    NUM_CLASSES = 6
    CLASS_NAMES = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]

    # =========================================================================
    # Training Parameters
    # =========================================================================
    BATCH_SIZE = 32  # Suitable for A100 40GB with this dual architecture
    EPOCHS = 2  # Limited epochs as per strategy (train on full data)

    # Optimizer & Scheduler
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    MAX_LR = 5e-3  # For OneCycleLR
    PCT_START = 0.3  # Percentage of training to increase LR

    # Loss
    LABEL_SMOOTHING = (
        0.0  # Using KL Divergence on soft targets, so smoothing is implicit
    )

    # Early Stopping
    PATIENCE = 3
    MIN_DELTA = 0.0001

    # Debugging / Development
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SUBSET_SIZE = 1000

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        dirs = [
            cls.WORKING_DIR,
            cls.CHECKPOINT_DIR,
            cls.CACHE_DIR,
            os.path.dirname(cls.SUBMISSION_CSV),
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

        # Set deterministic behavior for PyTorch
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
