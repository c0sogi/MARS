import os


class Config:
    # -------------------------------------------------------------------------
    # File Paths and Directories
    # -------------------------------------------------------------------------
    # Read-only input directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Output directories (writable)
    # Using a specific subdirectory for this idea to avoid conflicts
    WORKING_DIR = "./working/idea_6"
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Model Artifacts
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    VOCAB_PATH = os.path.join(WORKING_DIR, "vocab.npy")

    # Submission Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Hyperparameters
    # -------------------------------------------------------------------------
    SEED = 42

    # Image Processing
    # Height is fixed to ensure consistent vertical feature density
    IMAGE_HEIGHT = 256
    # Width is dynamic (None) or padded to a max; specific handling in dataset
    INPUT_CHANNELS = 1  # Grayscale conversion as per baseline idea

    # Text Processing
    # Max length observed in EDA was 403; adding buffer
    MAX_TEXT_LENGTH = 450

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    # Backbone for visual feature extraction
    ENCODER_NAME = "resnet18"
    ENCODER_PRETRAINED = True

    # 1D CNN / Gating parameters
    DECODER_CHANNELS = 512
    DROPOUT = 0.2

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler parameters
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 2

    # Hardware
    NUM_WORKERS = 8  # Optimized for 12 vCPUs

    # Optimization
    PATIENCE = 5  # Early stopping patience
    CLIP_GRAD = 5.0  # Gradient clipping value

    # -------------------------------------------------------------------------
    # Debugging / Development
    # -------------------------------------------------------------------------
    # Set to True to train on a small subset of data for quick checks
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 5000

    @staticmethod
    def create_directories():
        """
        Creates necessary working and submission directories.
        Should be called at the start of any script using this config.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
