import os
import torch


class Config:
    # System
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Using available vCPUs

    # File Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_13"
    SUBMISSION_DIR = "./submission"

    # Specific File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "cores2net_unet_best.pth")

    # Data Hyperparameters
    PATCH_SIZE = 128
    PATCHES_PER_IMAGE = 100  # High-density sampling

    # Training Hyperparameters
    NUM_EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # Strong regularization

    # Model Hyperparameters
    BASE_FILTERS = 64
    RES2NET_SCALE = 4  # Number of feature groups in Res2Net block

    @staticmethod
    def setup_directories():
        """
        Ensures that necessary working and submission directories exist.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        print(f"Directories ensured: {Config.WORKING_DIR}, {Config.SUBMISSION_DIR}")


# Execute setup immediately when module is imported to ensure safety
Config.setup_directories()
