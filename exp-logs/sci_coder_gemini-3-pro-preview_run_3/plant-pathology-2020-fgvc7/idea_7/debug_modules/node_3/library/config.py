import os
import torch


class Config:
    """
    Configuration class for Apple Disease Detection.
    Centralizes all hyperparameters, file paths, and model configurations
    for the Hybrid Ensemble (EfficientNet-B4 + Swin-Small) approach.
    """

    # ====================================================
    # General Settings
    # ====================================================
    SEED = 42
    DEBUG = False  # Set to True to run with a small subset of data for debugging
    DEBUG_SAMPLE_SIZE = 50  # Number of samples to use if DEBUG is True

    # ====================================================
    # Directory Paths
    # ====================================================
    INPUT_DIR = "./input"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory (Idea 7 specific)
    WORKING_DIR = "./working/idea_7"

    # Submission Directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ====================================================
    # Data & Caching
    # ====================================================
    # Path to save/load cached class weights (numpy format)
    CLASS_WEIGHTS_PATH = os.path.join(WORKING_DIR, "class_weights.npy")

    # Class definitions
    NUM_CLASSES = 4
    CLASS_LABELS = ["healthy", "multiple_diseases", "rust", "scab"]

    # ====================================================
    # Model Architectures
    # ====================================================
    # Backbone 1: EfficientNet-B4
    # Resolution: 380x380 (High resolution for fine details)
    EFFNET_MODEL_NAME = "tf_efficientnet_b4_ns"
    EFFNET_IMG_SIZE = 380

    # Backbone 2: Swin Transformer Small
    # Resolution: 224x224 (Standard window size)
    SWIN_MODEL_NAME = "swin_small_patch4_window7_224"
    SWIN_IMG_SIZE = 224

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    N_FOLDS = 5
    EPOCHS = 25
    PATIENCE = 10  # Early stopping patience

    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    MIN_LR = 1e-6

    # Optimizer & Scheduler
    SCHEDULER_TYPE = "CosineAnnealingLR"
    T_MAX = EPOCHS  # For CosineAnnealing

    # ====================================================
    # Hardware
    # ====================================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Ensures necessary working directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
Config.setup()
