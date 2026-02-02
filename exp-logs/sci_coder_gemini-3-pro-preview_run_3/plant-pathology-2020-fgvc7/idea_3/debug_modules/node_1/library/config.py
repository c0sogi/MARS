import os
import torch
import random
import numpy as np


class Config:
    # ==== General Configuration ====
    SEED = 42
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use when DEBUG is True
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==== Paths ====
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Working directory for caching and checkpoints
    WORK_DIR = "./working/idea_3"

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==== Data ====
    # Class labels sorted alphabetically as per standard or metadata
    CLASSES = ["healthy", "multiple_diseases", "rust", "scab"]
    NUM_CLASSES = len(CLASSES)

    # Inverse Frequency Class Weights derived from metadata analysis
    # Counts: rust: 441, scab: 427, healthy: 374, multiple_diseases: 68
    # Total: 1310
    # Weight = Total / Count
    CLASS_WEIGHTS = torch.tensor(
        [
            1310 / 374,  # healthy
            1310 / 68,  # multiple_diseases (High weight)
            1310 / 441,  # rust
            1310 / 427,  # scab
        ],
        dtype=torch.float32,
    )

    # ==== Model 1: High-Resolution Expert (EfficientNet-B3) ====
    MODEL_1_NAME = "efficientnet_b3"
    MODEL_1_IMG_SIZE = 300
    # Progressive Resizing: Start small, increase to target size
    MODEL_1_START_IMG_SIZE = 224
    MODEL_1_RESIZE_EPOCH = 5  # Switch to 300x300 after this epoch

    # ==== Model 2: Contextual Expert (ConvNeXt-Tiny) ====
    MODEL_2_NAME = "convnext_tiny"
    MODEL_2_IMG_SIZE = 224

    # ==== Training Hyperparameters ====
    BATCH_SIZE = (
        16  # Adjust based on GPU VRAM (A100 40GB allows larger, but 16/32 is safe)
    )
    EPOCHS = 15
    LR = 1e-4
    WEIGHT_DECAY = 1e-4

    # Stochastic Weight Averaging (SWA)
    USE_SWA = True
    SWA_START_EPOCH = 10
    SWA_LR = 5e-5

    # ==== Augmentation & Inference ====
    # Strong geometric augmentations (ShiftScaleRotate, Flip)
    # NO Occlusion (Cutout, etc.)
    AUG_SCALE_LIMIT = 0.2
    AUG_ROTATE_LIMIT = 15
    AUG_SHIFT_LIMIT = 0.1

    # Test Time Augmentation
    USE_TTA = True  # Uses Original + HFlip + VFlip

    @classmethod
    def setup(cls):
        """Ensures necessary directories exist."""
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# Initialize directories on import
Config.setup()
