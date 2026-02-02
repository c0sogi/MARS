import os
import torch
import random
import numpy as np


class Config:
    # -------------------------------------------------------------------------
    # Project & Experiment Setup
    # -------------------------------------------------------------------------
    PROJECT_NAME = "idea_13"
    SEED = 42

    # -------------------------------------------------------------------------
    # Hardware & Runtime
    # -------------------------------------------------------------------------
    # Use 12 vCPUs and A100 GPU
    NUM_WORKERS = 12
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use Mixed Precision for A100 optimization
    USE_AMP = True

    # -------------------------------------------------------------------------
    # Paths / Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", PROJECT_NAME)

    # Metadata Sources
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Caching Paths (Parquet) - For deterministic data processing
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "processed_train.parquet")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "processed_val.parquet")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "processed_test.parquet")

    # Model & Submission Outputs
    # Note: The final submission must be at the root or specified location.
    # We save the model in the working directory.
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = "submission.csv"

    # -------------------------------------------------------------------------
    # Data Hyperparameters
    # -------------------------------------------------------------------------
    # Input Strategy: Early Fusion (Image + Age + Implant)
    IMG_SIZE = (768, 768)
    NUM_CHANNELS = 3  # 1 (Image) + 1 (Age Map) + 1 (Implant Map)

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    # Architecture: Attentive Pyramid Symmetry-Difference Siamese EfficientNet-B2
    BACKBONE = "efficientnet_b2"
    PRETRAINED = True
    DROP_RATE = 0.3
    DROP_PATH_RATE = 0.2

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # Batch Size: 12 fits comfortably on A100 40GB with 768x768 Siamese inputs
    BATCH_SIZE = 12
    NUM_EPOCHS = 10

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Loss: Weighted BCE to handle 1:47 imbalance
    POS_WEIGHT = 47.0

    # -------------------------------------------------------------------------
    # Debugging / Flexibility
    # -------------------------------------------------------------------------
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 500  # Subset size when DEBUG is True

    @classmethod
    def setup(cls):
        """
        Initializes the environment by creating necessary directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)

    @classmethod
    def override(cls, **kwargs):
        """
        Allows dynamic overriding of configuration parameters.
        Useful for tuning or debugging scripts.
        """
        for k, v in kwargs.items():
            if hasattr(cls, k):
                setattr(cls, k, v)


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Automatically setup directories on import
Config.setup()
