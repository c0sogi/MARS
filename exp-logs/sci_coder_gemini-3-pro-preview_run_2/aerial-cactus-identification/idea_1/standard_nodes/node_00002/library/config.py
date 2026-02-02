import os
import torch


class Config:
    """
    Global configuration for the Cactus Classification task.
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # File Paths
    # ==========================================
    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"

    # Specific project directory for caching/artifacts
    IDEA_DIR = os.path.join(WORKING_DIR, "idea_1")

    # Metadata files (pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Raw Data
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    # Note: Images are accessed via relative paths in metadata combined with INPUT_DIR

    # Output paths
    MODEL_SAVE_PATH = os.path.join(IDEA_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Parameters
    # ==========================================
    IMAGE_SIZE = (32, 32)
    IMAGE_HEIGHT = 32
    IMAGE_WIDTH = 32
    CHANNELS = 3

    # Normalization (Simple 0-1 scaling is often sufficient,
    # but standard ImageNet stats can be used if using transfer learning)
    # Here we stick to simple scaling logic handled in the dataset class.

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 35

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MIN_DELTA = 1e-4

    # ==========================================
    # Compute & Resources
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Number of workers for DataLoader.
    # 12 vCPUs available -> 4-8 workers is usually a good balance.
    NUM_WORKERS = 4

    # ==========================================
    # Debugging / Development
    # ==========================================
    # Set to True to train on a small subset of data for quick pipeline verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 500

    @classmethod
    def setup(cls):
        """
        Create necessary writable directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.IDEA_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Run setup immediately when module is imported to ensure directories exist
Config.setup()
