import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # Directory and File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata paths (generated previously)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching and model checkpoints
    # Optimization V1 specific directory
    WORKING_DIR = "./working/optimization_v1"
    CACHE_FILE_PATH = os.path.join(WORKING_DIR, "roi_cache_v2.parquet")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Processing Parameters
    # --------------------------------------------------------------------------
    IMG_SIZE = 224
    NUM_SLICES_PER_MODALITY = 3  # Anchor +/- 1 neighbor
    STRIDE = 5  # Fixed stride for neighbor selection

    # Modalities: FLAIR, T1w, T1wCE, T2w
    MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]
    NUM_MODALITIES = len(MODALITIES)

    # Input channels = 4 modalities * 3 slices = 12 channels
    IN_CHANNELS = NUM_MODALITIES * NUM_SLICES_PER_MODALITY

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------
    MODEL_NAME = "efficientnet_b0"
    NUM_CLASSES = 1
    DROPOUT_RATE = 0.3
    GROUPS = 4  # For the grouped convolutional stem

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    EARLY_STOPPING_PATIENCE = 5

    # Augmentation
    ROTATION_DEGREES = 15

    # --------------------------------------------------------------------------
    # Compute
    # --------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """
        Ensure necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set reproducible seeds
        import random
        import numpy as np

        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Initialize environment immediately upon import
Config.setup()
