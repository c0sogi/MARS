import os
import torch
import random
import numpy as np


def seed_everything(seed=42):
    """Sets the random seed for reproducibility across torch, numpy, and random."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    # --- Reproducibility ---
    SEED = 42

    # --- System ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Use fewer workers than max vCPUs to avoid overhead/contention
    NUM_WORKERS = 8

    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --- Model ---
    # Using ConvNeXt-Small as requested for better global context
    MODEL_NAME = "convnext_small"
    IMG_SIZE = 384
    NUM_CLASSES = 6
    # Alphabetical order is standard for MultiLabelBinarizer
    CLASSES = [
        "complex",
        "frog_eye_leaf_spot",
        "healthy",
        "powdery_mildew",
        "rust",
        "scab",
    ]

    # --- Training Hyperparameters ---
    DEBUG = False
    DEBUG_SUBSET_SIZE = 1000

    EPOCHS = 15
    BATCH_SIZE = 32  # Adjusted for 384x384 image size and GPU memory

    # Optimizer (AdamW)
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    MIN_LR = 1e-6

    # --- Regularization (Mixup & CutMix) ---
    # Explicitly enabled to prevent overfitting on larger resolution
    MIXUP_ENABLED = True
    MIXUP_ALPHA = 0.4
    CUTMIX_ALPHA = 1.0
    MIXUP_PROB = 0.5  # Probability of applying Mixup/CutMix batch transformation

    # --- Compute ---
    USE_AMP = True  # Automatic Mixed Precision

    # --- Inference ---
    THRESHOLD = 0.5

    @classmethod
    def print_config(cls):
        print(f"\n{'='*20} Configuration {'='*20}")
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k}: {v}")
        print(f"{'='*55}\n")


# Apply seed immediately upon import
seed_everything(Config.SEED)
