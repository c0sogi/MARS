import os


class Config:
    """
    Configuration for the Right Whale Detection Task.
    Implements the 'Heterogeneous Ensemble of Physically-Aligned Legacy Architectures' strategy.
    """

    # ==========================================
    # Project & Paths
    # ==========================================
    PROJECT_NAME = "idea_11"
    SEED = 42

    # Input Directories (Read-Only)
    INPUT_ROOT = "./input"
    TRAIN_DIR = os.path.join(INPUT_ROOT, "train2")
    TEST_DIR = os.path.join(INPUT_ROOT, "test2")

    # Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_ROOT, "sampleSubmission.csv")

    # Output Directories (Writeable)
    WORKING_DIR = "./working"
    OUTPUT_DIR = os.path.join(WORKING_DIR, PROJECT_NAME)
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Audio Processing (Physically-Aligned)
    # ==========================================
    SAMPLE_RATE = 2000
    # High FFT resolution for low-frequency whale calls
    N_FFT = 1024
    HOP_LENGTH = 64
    N_MELS = 128
    FMIN = 0
    FMAX = None  # Defaults to SR // 2

    # Data Normalization & Resizing
    # Use native resolution (approx 128x63) to avoid interpolation artifacts
    RESIZE_TO_IMAGENET = False
    # Instance Standardization: Zero-Mean, Unit-Variance per clip
    NORMALIZE_INSTANCE = True

    # ==========================================
    # Model Architecture
    # ==========================================
    # Heterogeneous Ensemble: EfficientNet-B0 + ResNet-34
    MODEL_NAMES = ["efficientnet_b0", "resnet34"]
    PRETRAINED = True
    IN_CHANNELS = 1
    NUM_CLASSES = 1
    # Generalized Mean Pooling to detect transient calls
    POOLING_TYPE = "gem"

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    FOLDS = 5
    EPOCHS = 20
    BATCH_SIZE = 128

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    OPTIMIZER = "AdamW"

    # Scheduler: Cosine Annealing
    SCHEDULER = "CosineAnnealingLR"
    MIN_LR = 1e-6
    T_MAX = EPOCHS  # For CosineAnnealing

    # Regularization & Early Stopping
    PATIENCE = 7

    # ==========================================
    # Augmentation
    # ==========================================
    # SpecAugment enabled, Mixup explicitly disabled
    USE_SPECAUG = True
    MASK_TIME_PARAM = 10
    MASK_FREQ_PARAM = 10
    USE_MIXUP = False

    # ==========================================
    # Runtime & Debugging
    # ==========================================
    NUM_WORKERS = 8
    DEBUG = False
    DEBUG_SUBSET_SIZE = 500  # Number of samples for debugging

    @classmethod
    def setup(cls):
        """Creates necessary output directories."""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Ensure directories exist upon import
Config.setup()
