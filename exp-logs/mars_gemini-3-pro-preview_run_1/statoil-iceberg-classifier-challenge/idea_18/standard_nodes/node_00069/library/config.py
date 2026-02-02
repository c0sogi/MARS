import os


class Config:
    # ==========================================
    # Directories and Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching and intermediate files
    # Using 'idea_18' as the specific experiment identifier
    WORKING_DIR = "./working/idea_18"
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Raw Data Paths
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # ==========================================
    # Global Statistics (from Data Analysis)
    # ==========================================
    # Used for Global Min-Max Normalization
    BAND1_MIN = -45.5944
    BAND1_MAX = 32.1806
    BAND2_MIN = -45.6555
    BAND2_MAX = 17.8628

    # Used for Inc Angle Normalization (Standard Scaling)
    INC_ANGLE_MEAN = 39.2829
    INC_ANGLE_STD = 3.8362

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    IMG_SIZE = 224  # Upsampling target (Bicubic)
    N_CHANNELS = 3  # Band 1, Band 2, Average
    NUM_CLASSES = 1  # Binary classification

    # Augmentation
    ROTATION_DEGREES = 20  # Continuous rotation +/- 20 degrees

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    BACKBONE = "resnet18"
    PRETRAINED = True

    # Adaptive GeM Pooling
    GEM_P_INIT = 1.0  # Initialize as Average Pooling
    GEM_EPS = 1e-6  # Numerical stability

    # Head
    DROPOUT_RATE = 0.5

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # Optimization
    LEARNING_RATE = 1e-4  # Initial LR for AdamW
    WEIGHT_DECAY = 0.01
    LABEL_SMOOTHING = 0.05

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 3
    MIN_LR = 1e-6

    # Phase 1: Calibration (5-Fold CV)
    N_FOLDS = 5
    PHASE1_MAX_EPOCHS = 50  # Upper bound to find convergence
    EARLY_STOPPING_PATIENCE = 10

    # Phase 2: Production (Full-Fit SWA)
    # Note: The base training duration for Phase 2 is determined dynamically
    # by the optimal epoch found in Phase 1.
    SWA_DURATION = 12  # Number of epochs for SWA
    SWA_LR = 1e-5  # Constant LR for SWA phase

    # ==========================================
    # Inference
    # ==========================================
    # TTA Options: Original, Horizontal Flip, Vertical Flip
    TTA_FLIPS = ["none", "horizontal", "vertical"]

    @classmethod
    def setup(cls):
        """Ensure necessary directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Run setup immediately when module is imported to ensure paths exist
Config.setup()
