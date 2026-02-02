import os


class Config:
    """
    Configuration class for the Ship vs Iceberg classification task.
    Centralizes file paths, hyperparameters, and constants.
    """

    def __init__(self):
        # ==========================================
        # Directory Paths
        # ==========================================
        self.INPUT_DIR = "./input"
        self.METADATA_DIR = "./metadata"
        self.WORKING_DIR = "./working/idea_1"
        self.SUBMISSION_DIR = "./submission"

        # Ensure working and submission directories exist
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)

        # ==========================================
        # File Paths
        # ==========================================
        # Raw Data
        self.TRAIN_JSON = os.path.join(self.INPUT_DIR, "train.json")
        self.TEST_JSON = os.path.join(self.INPUT_DIR, "test.json")
        self.SAMPLE_SUBMISSION = os.path.join(self.INPUT_DIR, "sample_submission.csv")

        # Metadata
        self.TRAIN_META_PATH = os.path.join(self.METADATA_DIR, "train_metadata.csv")
        self.VAL_META_PATH = os.path.join(self.METADATA_DIR, "val_metadata.csv")
        self.TEST_META_PATH = os.path.join(self.METADATA_DIR, "test_metadata.csv")

        # Cache Files (Parquet/NPY)
        self.CACHE_TRAIN_DATA = os.path.join(self.WORKING_DIR, "train_data.npy")
        self.CACHE_TRAIN_LABELS = os.path.join(self.WORKING_DIR, "train_labels.npy")
        self.CACHE_TRAIN_ANGLES = os.path.join(self.WORKING_DIR, "train_angles.npy")

        self.CACHE_VAL_DATA = os.path.join(self.WORKING_DIR, "val_data.npy")
        self.CACHE_VAL_LABELS = os.path.join(self.WORKING_DIR, "val_labels.npy")
        self.CACHE_VAL_ANGLES = os.path.join(self.WORKING_DIR, "val_angles.npy")

        self.CACHE_TEST_DATA = os.path.join(self.WORKING_DIR, "test_data.npy")
        self.CACHE_TEST_IDS = os.path.join(self.WORKING_DIR, "test_ids.npy")
        self.CACHE_TEST_ANGLES = os.path.join(self.WORKING_DIR, "test_angles.npy")

        # Output
        self.SUBMISSION_FILE = os.path.join(self.SUBMISSION_DIR, "submission.csv")
        self.MODEL_CHECKPOINT = os.path.join(self.WORKING_DIR, "best_model.pth")

        # ==========================================
        # Data Parameters
        # ==========================================
        self.IMG_WIDTH = 75
        self.IMG_HEIGHT = 75
        self.IMG_CHANNELS = 3  # Band 1, Band 2, Mean(B1, B2)

        # Normalization Constants (dB)
        # Based on analysis: Min ~ -45, Max ~ 32.
        # We use slightly wider bounds to ensure values stay roughly within [0, 1]
        self.MIN_DB = -50.0
        self.MAX_DB = 40.0

        # Imputation for Incidence Angle
        # Based on training data analysis mean
        self.INC_ANGLE_MEAN = 39.2829

        # ==========================================
        # Training Hyperparameters
        # ==========================================
        self.SEED = 42
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 0.001
        self.NUM_EPOCHS = 50
        self.PATIENCE = 10  # For Early Stopping

        # ==========================================
        # Model Architecture Hyperparameters
        # ==========================================
        self.DROPOUT_RATE = 0.5
        self.CONV_FILTERS = [64, 128, 128]
        self.DENSE_UNITS = 512
