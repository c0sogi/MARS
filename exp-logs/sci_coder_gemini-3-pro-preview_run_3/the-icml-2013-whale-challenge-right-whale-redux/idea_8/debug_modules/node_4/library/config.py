import os
import torch


class Config:
    """
    Central configuration for the Right Whale Detection task.
    Encapsulates file paths, model hyperparameters, training settings,
    and data processing configurations.
    """

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train2")
    TEST_DIR = os.path.join(INPUT_DIR, "test2")

    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for this experiment (Idea 8)
    WORKING_DIR = "./working/idea_8"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Data Parameters
    # =========================================================================
    SR = 2000  # Sampling rate (based on dataset analysis)
    DURATION = 2.0  # Clip duration in seconds
    N_MELS = 224  # Number of Mel bands (High resolution)
    N_FFT = 512  # FFT window size (~256ms)
    HOP_LENGTH = 20  # Hop length (10ms)
    FMIN = 10  # Minimum frequency
    FMAX = 1000  # Maximum frequency (Nyquist at 2000Hz SR)

    # Input image size for the model (Height, Width)
    # Height = N_MELS, Width = Time steps (approx 200 for 2s @ 10ms hop)
    # We resize to square for standard CNN backbones
    IMAGE_SIZE = (224, 224)

    # Normalization
    FREQ_WISE_NORM = (
        True  # Normalize each frequency bin independently (Stationary noise removal)
    )

    # =========================================================================
    # Model Architecture
    # =========================================================================
    BACKBONE = "convnext_tiny.fb_in1k"
    PRETRAINED = True
    IN_CHANNELS = 1  # Spectrograms are 1-channel
    NUM_CLASSES = 1  # Binary classification (Whale vs Noise)

    # Custom Modules
    USE_COORDINATE_ATTENTION = True  # Factorized attention for time/freq localization
    USE_GEM_POOL = True  # Generalized Mean Pooling for weak supervision

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Training loop settings
    EPOCHS = 20
    BATCH_SIZE = 64
    NUM_WORKERS = 4

    # Debugging
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SUBSET_SIZE = 500  # Number of samples to use in debug mode

    # Optimization
    LEARNING_RATE = 1e-3
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-2
    SCHEDULER = "CosineAnnealingLR"  # Cosine Annealing

    # Loss Function & Class Imbalance
    USE_WEIGHTED_LOSS = True  # Weight BCE by inverse class frequency

    # Augmentation
    MIXUP_ALPHA = 0.4  # Mixup strength (0.0 to disable)
    USE_SPECAUG = True  # Apply Time/Freq masking

    # SpecAugment settings
    MASK_TIME_PROB = 0.1
    MASK_FREQ_PROB = 0.1
    MASK_TIME_LENGTH = 15
    MASK_FREQ_LENGTH = 15

    # =========================================================================
    # Inference
    # =========================================================================
    # Whether to reload the best checkpoint (based on Val AUC) before inference
    RELOAD_BEST_MODEL = True

    @classmethod
    def display(cls):
        """Prints the configuration."""
        print("\n" + "=" * 40)
        print(f"{'CONFIGURATION':^40}")
        print("=" * 40)
        for key, value in cls.__dict__.items():
            if not key.startswith("__") and not callable(value):
                print(f"{key:<25} : {value}")
        print("=" * 40 + "\n")
