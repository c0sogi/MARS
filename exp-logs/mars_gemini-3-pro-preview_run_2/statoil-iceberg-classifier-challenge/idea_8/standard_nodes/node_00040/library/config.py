import os


class Config:
    """
    Configuration class for the Multi-Scale Attention Hybrid Network (MSA-HN) project.
    Centralizes all file paths, hyperparameters, and constants.
    """

    # ==========================================
    # 1. GLOBAL SETTINGS
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to use a subset of data for quick debugging
    SUBSET_SIZE = 200  # Number of samples to use when DEBUG is True

    # ==========================================
    # 2. FILE PATHS
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_8"
    SUBMISSION_DIR = "./submission"

    # Raw Data
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata Splits
    TRAIN_META_FILE = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_FILE = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_FILE = os.path.join(METADATA_DIR, "test.csv")

    # Outputs
    PROCESSED_DATA_FILE = os.path.join(WORKING_DIR, "processed_data.npz")
    MODEL_CHECKPOINT_PREFIX = os.path.join(WORKING_DIR, "msa_hn_model_fold")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 3. DATA PARAMETERS
    # ==========================================
    IMG_HEIGHT = 75
    IMG_WIDTH = 75
    # 3 Channels: Band 1 (HH), Band 2 (HV), Average ((HH+HV)/2)
    IMG_CHANNELS = 3

    # ==========================================
    # 4. MODEL HYPERPARAMETERS
    # ==========================================
    DROPOUT_RATE = 0.2

    # ==========================================
    # 5. TRAINING HYPERPARAMETERS
    # ==========================================
    N_FOLDS = 5
    BATCH_SIZE = 32
    EPOCHS = 100  # "Low and Slow" strategy implies sufficient epochs

    # Optimization
    LEARNING_RATE = 2e-4  # Conservative start
    WEIGHT_DECAY = 1e-4  # To be applied conditionally

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 15

    @classmethod
    def setup(cls):
        """
        Ensures that the necessary working and submission directories exist.
        Should be called at the start of execution.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
