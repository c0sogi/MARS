import os


class Config:
    """
    Configuration parameters for the Iceberg Classifier project.
    Centralizes file paths, hyperparameters, and model constants.
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # File System Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"

    # Cache directory for this specific idea (Idea 2: D2N Improved)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_2")

    # Input Data Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata Files (Pre-split)
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    MODEL_CHECKPOINT = os.path.join(CACHE_DIR, "d2n_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing Parameters
    # ==========================================
    # Target size for downsampling (48x48)
    IMAGE_SIZE = 48
    NUM_BANDS = 2

    # Flattened image dimension: 32 * 32 * 2 = 2048
    FLATTENED_IMAGE_DIM = IMAGE_SIZE * IMAGE_SIZE * NUM_BANDS

    # Total input dimension for MLP: Flattened Image + 1 (Incidence Angle)
    INPUT_DIM = FLATTENED_IMAGE_DIM + 1

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Dense Network Architecture
    HIDDEN_UNITS = [512, 256]
    DROPOUT_RATE = 0.5

    # Training Loop
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0002
    NUM_EPOCHS = 50
    PATIENCE = 10  # For Early Stopping

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories if they do not exist.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
