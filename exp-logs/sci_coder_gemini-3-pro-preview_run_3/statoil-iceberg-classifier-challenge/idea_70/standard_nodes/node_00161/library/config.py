import os


class Config:
    # ==========================================
    # Path Configurations
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Cache directory specific to this idea/experiment
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_70")
    CHECKPOINT_DIR = os.path.join(CACHE_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Raw Data Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Pre-generated)
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # ==========================================
    # Data Parameters
    # ==========================================
    IMG_HEIGHT = 75
    IMG_WIDTH = 75
    # 3 Channels: HH, HV, and Synthetic Average ((HH+HV)/2)
    CHANNELS = 3

    # ==========================================
    # Model Parameters
    # ==========================================
    SEED = 42
    NUM_FOLDS = 5

    # Architecture specific
    ATTENTION_REDUCTION_RATIO = 16
    FEATURE_DIM = 256  # Dimension after isomorphic readout

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3  # Constant LR as per strategy
    NUM_EPOCHS = 75
    PATIENCE = 12
    WEIGHT_DECAY = 1e-4  # For AdamW
    DROPOUT_RATE = 0.5

    # ==========================================
    # Runtime / Debugging
    # ==========================================
    # Set to True to run on a small subset for debugging pipeline
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100

    # Inference
    USE_TTA = False  # Explicitly disabled per strategy

    @classmethod
    def setup(cls):
        """
        Creates the necessary working directories for cache, checkpoints,
        and submissions.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
