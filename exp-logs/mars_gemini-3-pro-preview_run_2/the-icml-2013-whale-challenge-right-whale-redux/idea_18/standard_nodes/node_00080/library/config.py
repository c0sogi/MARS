import os


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    PROJECT_NAME = "RightWhaleDetection"
    IDEA_NAME = "idea_18"
    SEED = 42
    DEBUG = False  # Set to True to use a subset of data for debugging

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    # Input Directories (Read-Only)
    INPUT_ROOT = "./input"
    TRAIN_DIR = os.path.join(INPUT_ROOT, "train2")
    TEST_DIR = os.path.join(INPUT_ROOT, "test2")

    # Metadata Paths (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_ROOT, "sampleSubmission.csv")

    # Working Directory (Write Access)
    WORKING_DIR = os.path.join("./working", IDEA_NAME)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Audio / Spectrogram Parameters
    # =========================================================================
    SAMPLE_RATE = 2000  # Native sample rate of the dataset
    N_FFT = 1024  # High frequency resolution
    HOP_LENGTH = 64  # High time resolution
    N_MELS = 128  # Number of Mel bands
    F_MIN = 0  # Minimum frequency
    F_MAX = None  # Maximum frequency (None = SR/2)
    TOP_DB = 80.0  # Dynamic range for dB conversion (clamping noise floor)
    NORMALIZED = False  # Preserve environmental spectral tilt

    # Input Dimensions (Implicitly defined by SR, Duration, Hop, N_Mels)
    # Duration is ~2s. 2000 samples / 64 hop ~ 31 frames per second.
    # Total frames approx 63-64.
    # Input shape to model will be [Batch, 1, 128, ~63] (Frequency, Time)

    # =========================================================================
    # Data Preprocessing & Augmentation
    # =========================================================================
    RESIZE_SPECTROGRAM = False  # Use native resolution (do not resize to 224x224)
    INSTANCE_NORM = True  # Zero-Mean, Unit-Variance per clip

    # SpecAugment
    FREQ_MASK_PARAM = 20
    TIME_MASK_PARAM = 0  # Usually less critical for short clips, but can be added
    MASK_MODE = "mean"  # Mask with mean value

    USE_MIXUP = False  # Mixup excluded per strategy

    # =========================================================================
    # Model Architecture
    # =========================================================================
    IN_CHANNELS = 1
    POOLING_TYPE = "gem"  # Generalized Mean Pooling
    PRETRAINED = True

    # Architectures
    ARCH_EFFICIENTNET = "tf_efficientnet_b0_ns"  # Noisy Student weights
    ARCH_RESNET = "resnet34"  # ImageNet weights

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    N_FOLDS = 5
    BATCH_SIZE = 128
    EPOCHS = 30  # Sufficient for convergence with Early Stopping
    PATIENCE = 5  # Strict patience

    LEARNING_RATE = 1e-3  # Base LR
    WEIGHT_DECAY = 1e-4  # Low weight decay for pre-trained models
    MIN_LR = 1e-6

    NUM_WORKERS = 4  # CPU workers for dataloading

    # =========================================================================
    # Ensemble Configuration Matrix
    # =========================================================================
    # Defines the 4 configurations for Level 0 Base Learners
    # Keys:
    #   'arch': Model architecture name
    #   'objective': Convergence criterion ('auc' or 'loss')
    #   'name': Unique identifier for saving checkpoints

    ENSEMBLE_CONFIGS = [
        {"arch": ARCH_EFFICIENTNET, "objective": "auc", "name": "effnet_b0_auc"},
        {"arch": ARCH_EFFICIENTNET, "objective": "loss", "name": "effnet_b0_loss"},
        {"arch": ARCH_RESNET, "objective": "auc", "name": "resnet34_auc"},
        {"arch": ARCH_RESNET, "objective": "loss", "name": "resnet34_loss"},
    ]

    # Level 1 Meta-Learner
    META_LEARNER_MODEL = "LogisticRegression"
