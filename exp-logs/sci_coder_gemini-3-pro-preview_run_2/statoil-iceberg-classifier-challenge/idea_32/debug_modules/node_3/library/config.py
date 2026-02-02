import os


class Config:
    """
    Configuration for the Tri-Path Wide-Body Network (TP-WBN) solution.
    Defines paths, training hyperparameters, model settings, and reproducibility seeds.
    """

    # ==========================================
    # 1. PATHS & DIRECTORIES
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORK_DIR = "./working/idea_32"
    CACHE_DIR = os.path.join(WORK_DIR, "cache")

    # Input Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(WORK_DIR, "submission.csv")
    PROCESSED_DATA_PATH = os.path.join(CACHE_DIR, "processed_data.npz")

    # ==========================================
    # 2. REPRODUCIBILITY
    # ==========================================
    SEED = 42

    # ==========================================
    # 3. DATA CONFIGURATION
    # ==========================================
    IMAGE_SIZE = 75
    INPUT_CHANNELS = 3  # Band 1, Band 2, Mean(Band 1, Band 2)

    # Augmentation
    ROTATION_ANGLES = [0, 90, 180, 270]

    # ==========================================
    # 4. MODEL HYPERPARAMETERS
    # ==========================================
    # Wide-Body Backbone
    BACKBONE_FILTERS = 128

    # Regularization
    DROPOUT_RATE = 0.5

    # ==========================================
    # 5. TRAINING HYPERPARAMETERS
    # ==========================================
    BATCH_SIZE = 32
    EPOCHS = 100
    PATIENCE = 10  # For Early Stopping
    LEARNING_RATE = 1e-3  # Adam Optimizer default
    NUM_FOLDS = 5  # Stratified K-Fold

    @classmethod
    def setup(cls):
        """
        Creates the necessary working and cache directories.
        """
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        print(f"Directories ensured: {cls.WORK_DIR}, {cls.CACHE_DIR}")

    @classmethod
    def get_model_path(cls, fold_idx):
        """
        Returns the path for saving/loading the model checkpoint for a specific fold.
        """
        return os.path.join(cls.WORK_DIR, f"tp_wbn_fold_{fold_idx}.pth")
