import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    PROJECT_NAME = "HarmfulBrainActivityDetection"
    IDEA_NAME = "idea_12"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SIZE = 100  # Number of samples to use in debug mode

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", IDEA_NAME)

    # Input Data Paths
    TRAIN_EEGS_DIR = os.path.join(INPUT_DIR, "train_eegs")
    TRAIN_SPECTROGRAMS_DIR = os.path.join(INPUT_DIR, "train_spectrograms")
    TEST_EEGS_DIR = os.path.join(INPUT_DIR, "test_eegs")
    TEST_SPECTROGRAMS_DIR = os.path.join(INPUT_DIR, "test_spectrograms")

    # Metadata Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    # =========================================================================
    # Data Parameters
    # =========================================================================
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

    # EEG Signal
    EEG_RAW_SR = 200  # Original sampling rate
    EEG_TARGET_SR = 100  # Downsampled rate (Nyquist safe)
    EEG_DURATION = 50  # Seconds
    EEG_SEQ_LEN = EEG_TARGET_SR * EEG_DURATION  # 5000 time steps
    EEG_CHANNELS = 20  # 19 EEG + 1 EKG

    # Spectrogram
    SPEC_DURATION = 600  # 10 minutes (600 seconds)
    SPEC_SIZE = (512, 512)  # (Height, Width) for resizing
    SPEC_CHANNELS = 5  # LL, RL, LP, RP + Coordinate Map

    # =========================================================================
    # Model Architecture
    # =========================================================================
    MODEL_NAME = "AsymmetricCoordinateTransformer"

    # Stream B: Spectrogram Encoder
    BACKBONE_2D = "efficientnet_b1"
    PRETRAINED_2D = True

    # Stream A: EEG Encoder (1D CNN)
    EEG_KERNELS = [3, 5, 7, 9]  # Multi-scale kernels
    EEG_FILTERS = [64, 128, 256, 512]

    # Fusion: Transformer Decoder
    D_MODEL = 256  # Embedding dimension for transformer
    NHEAD = 8  # Number of attention heads
    NUM_DECODER_LAYERS = 2  # Number of decoder layers
    DIM_FEEDFORWARD = 1024  # FFN dimension
    DROPOUT = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32  # Optimized for A100 40GB
    EPOCHS = 2  # Limited epochs as per strategy

    # Optimizer (AdamW)
    LR = 1e-3
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1.0

    # Scheduler (OneCycleLR)
    PCT_START = 0.3
    DIV_FACTOR = 25.0
    FINAL_DIV_FACTOR = 1000.0

    # Loss
    LOSS_FUNCTION = "KLDivLoss"  # Kullback-Leibler Divergence

    # =========================================================================
    # Compute & Environment
    # =========================================================================
    NUM_WORKERS = 4  # 12 vCPUs available
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP = True  # Mixed Precision Training (FP16)

    @classmethod
    def setup(cls):
        """
        Initialize the working directories.
        This ensures the path ./working/idea_12 exists before any processing.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)

        # Set deterministic behavior for PyTorch (basic level)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)


# Execute setup on module import
Config.setup()
