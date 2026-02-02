import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # General Setup
    # --------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Adjust based on available vCPUs (12 available)

    # --------------------------------------------------------------------------
    # Data Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata
    METADATA_DIR = "./metadata"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory (Idea Specific)
    WORKING_DIR = "./working/idea_34"
    CACHE_DIR = WORKING_DIR  # Using the working dir for caching parquet/npy files

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Processing & ROI Selection
    # --------------------------------------------------------------------------
    IMG_SIZE = (224, 224)
    NUM_MODALITIES = 4  # FLAIR, T1w, T1wCE, T2w
    STACK_DEPTH = 3  # [ID-5, ID, ID+5]
    NUM_CHANNELS = NUM_MODALITIES * STACK_DEPTH  # 12 Channels

    # Fidelity-Aligned ROI Parameters
    ROI_STRIDE = 5
    ROI_SEARCH_START = 0.15
    ROI_SEARCH_END = 0.85

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------
    BACKBONE = "efficientnet_v2_s"
    PRETRAINED = True
    DROP_RATE = 0.5
    NUM_CLASSES = 1

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    PATIENCE = 5  # For Early Stopping

    # --------------------------------------------------------------------------
    # Augmentation
    # --------------------------------------------------------------------------
    ROTATION_DEGREES = 15  # +/- 15 degrees

    @classmethod
    def setup(cls):
        """Creates necessary output directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories immediately when config is imported
Config.setup()
