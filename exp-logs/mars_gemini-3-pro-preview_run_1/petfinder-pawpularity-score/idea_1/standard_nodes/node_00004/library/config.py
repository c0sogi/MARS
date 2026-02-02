import os
import torch


class Config:
    """
    Global configuration for the Pawpularity prediction task.
    """

    # ==========================================
    # System & Reproducibility
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Adjust based on available vCPUs (12 available)

    # ==========================================
    # Hyperparameters
    # ==========================================
    IMG_SIZE = 224
    BATCH_SIZE = 64
    EPOCHS = 50  # Max epochs, though early stopping should trigger sooner
    LEARNING_RATE = 1e-3

    # Model Architecture
    MODEL_NAME = "tf_mobilenetv3_large_100"

    # Regularization for Ridge Regression (if applicable)
    RIDGE_ALPHA = 1.0

    # ==========================================
    # Debugging
    # ==========================================
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100  # Number of samples to use when DEBUG is True

    # ==========================================
    # Paths
    # ==========================================
    # Root directories
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"

    # Idea-specific working directory (for caching)
    IDEA_DIR = os.path.join(WORKING_DIR, "idea_1")

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Submission File
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Caching Paths for Linear Probing (Feature Extraction)
    # We use .npy for efficient numpy array storage
    CACHE_TRAIN_FEATURES = os.path.join(IDEA_DIR, f"{MODEL_NAME}_train_features.npy")
    CACHE_TRAIN_TARGETS = os.path.join(IDEA_DIR, f"{MODEL_NAME}_train_targets.npy")
    CACHE_VAL_FEATURES = os.path.join(IDEA_DIR, f"{MODEL_NAME}_val_features.npy")
    CACHE_VAL_TARGETS = os.path.join(IDEA_DIR, f"{MODEL_NAME}_val_targets.npy")
    CACHE_TEST_FEATURES = os.path.join(IDEA_DIR, f"{MODEL_NAME}_test_features.npy")
    CACHE_TEST_IDS = os.path.join(IDEA_DIR, f"{MODEL_NAME}_test_ids.npy")

    # ==========================================
    # Setup
    # ==========================================
    @classmethod
    def setup(cls):
        """
        Ensures necessary writable directories exist.
        """
        os.makedirs(cls.IDEA_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Run setup immediately when module is imported
Config.setup()
