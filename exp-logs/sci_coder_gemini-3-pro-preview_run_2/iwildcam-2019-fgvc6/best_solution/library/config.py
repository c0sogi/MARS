import os
import torch
import numpy as np
import random


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    """
    Central configuration for the EfficientNet-B4 Native Resolution pipeline.
    """

    # ==========================================
    # Directories and Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    # Native resolution for EfficientNet-B4 is 380x380
    IMAGE_SIZE = 380
    NUM_CLASSES = 23

    # Compute resources
    NUM_WORKERS = 8  # 12 vCPUs available, leaving some overhead

    # Debugging / Development
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 2000  # Number of images to use if DEBUG is True

    # ==========================================
    # Model Configuration
    # ==========================================
    MODEL_NAME = "tf_efficientnet_b4"
    PRETRAINED = True

    # ==========================================
    # Training Configuration
    # ==========================================
    SEED = 42

    # Batch Size Strategy
    # A100 40GB allows for decent batch size even with 380px images.
    # We use Gradient Accumulation to simulate larger batches if needed.
    BATCH_SIZE = 32
    GRAD_ACCUM_STEPS = 1  # Effective Batch Size = BATCH_SIZE * GRAD_ACCUM_STEPS

    # Stage 1: Head Alignment (Frozen Backbone)
    # High LR to quickly adapt the random head to the pre-trained features
    LR_STAGE1 = 1e-3
    EPOCHS_STAGE1 = 5

    # Stage 2: Fine-Tuning (Unfrozen Backbone)
    # Lower LR to gently adjust the backbone weights
    LR_STAGE2 = 1e-4
    EPOCHS_STAGE2 = 10

    # Regularization
    WEIGHT_DECAY = 1e-4

    # ==========================================
    # Device
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print(f"{'='*30}")
        print(f"CONFIGURATION: {cls.MODEL_NAME}")
        print(f"{'='*30}")
        for key, value in cls.__dict__.items():
            if not key.startswith("__") and not callable(value):
                print(f"{key}: {value}")
        print(f"{'='*30}")
