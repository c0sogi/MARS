import os


class Config:
    """
    Configuration for the Hybrid Neuro-Symbolic Ensemble strategy.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    NUM_CLASSES = 19
    N_FOLDS = 5
    NUM_WORKERS = 2  # Adjusted for available vCPUs

    # =========================================================================
    # File Paths
    # =========================================================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Input Data Sources
    # Stream A: CNNs use Standard Spectrograms
    SPECTROGRAM_DIR = os.path.join(INPUT_ROOT, "supplemental_data", "spectrograms")
    # Stream B: MLP uses Histogram of Segments (BoAW)
    HISTOGRAM_FILE = os.path.join(
        INPUT_ROOT, "supplemental_data", "histogram_of_segments.txt"
    )

    # Metadata Files (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories
    # Using specific idea folder for caching and checkpoints
    WORKING_DIR = "./working/idea_29"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Preprocessing & Augmentation
    # =========================================================================
    # CNN Stream
    IMG_SIZE = (224, 224)
    IMG_CHANNELS = 3  # Replicate single channel to 3 for pre-trained models

    # Augmentation Constraints
    AUG_SHIFT_LIMIT = 0.1  # <10% width translation (Safe-Zone)
    AUG_BRIGHTNESS_LIMIT = 0.2  # Photometric Jitter
    AUG_CONTRAST_LIMIT = 0.2  # Photometric Jitter
    USE_HORIZONTAL_FLIP = False  # Strictly disabled

    # Regularization
    MIXUP_ALPHA = 0.4  # Bias towards ground truth

    # MLP Stream
    MLP_FEATURE_NOISE = 0.01  # Gaussian noise std dev for feature augmentation

    # =========================================================================
    # Model Architectures
    # =========================================================================
    # Deep Stream (Texture Analysis)
    CNN_MODELS = ["resnet18", "efficientnet_b0", "densenet121"]

    # Symbolic Stream (Cluster Analysis)
    MLP_INPUT_DIM = 100
    MLP_HIDDEN_DIM = 128
    MLP_DROPOUT = 0.5

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    EPOCHS = 50  # Sufficient for patience to trigger
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # High weight decay for control
    PATIENCE = 15  # Aggressive early stopping

    # Snapshot Ensemble
    NUM_SNAPSHOTS = 3  # Save Top-3 best checkpoints per fold

    @classmethod
    def setup(cls):
        """
        Creates the necessary directory structure for artifacts.
        Should be called at the beginning of the pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Sub-directories for specific model checkpoints to keep things organized
        for model_name in cls.CNN_MODELS:
            os.makedirs(os.path.join(cls.CHECKPOINT_DIR, model_name), exist_ok=True)
        os.makedirs(os.path.join(cls.CHECKPOINT_DIR, "mlp"), exist_ok=True)
