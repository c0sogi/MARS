import os
import torch
import numpy as np
import random


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    # ==== Directories ====
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for Idea 5
    WORKING_DIR = "./working/idea_5"

    # Create working directory immediately
    os.makedirs(WORKING_DIR, exist_ok=True)

    # File Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")
    SUBMISSION_PATH = "./submission/submission.csv"

    # ==== Data ====
    CLASSES = ["healthy", "multiple_diseases", "rust", "scab"]
    NUM_CLASSES = len(CLASSES)
    LABEL_COLS = CLASSES

    # ==== Compute ====
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==== Training Hyperparameters ====
    N_FOLDS = 5
    EPOCHS = 30
    BATCH_SIZE = 16  # Reduced batch size to fit in 16GB VRAM
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 10  # Cite solution_lesson_node_00006

    # Loss scaling
    # We will compute inverse frequency weights dynamically in the training loop,
    # but we define the strategy here.
    USE_CLASS_WEIGHTS = True

    # ==== Model 1: EfficientNet-B4 ====
    # Using 'ns' (Noisy Student) weights for better performance
    MODEL_EFFNET = "tf_efficientnet_b4_ns"
    IMG_SIZE_EFFNET = 380

    # ==== Model 2: Swin Transformer Tiny ====
    # Window size 7 is standard for 224x224
    MODEL_SWIN = "swin_tiny_patch4_window7_224"
    IMG_SIZE_SWIN = 224

    # ==== Augmentation & TTA ====
    # TTA: Test Time Augmentation (Original + Horizontal Flip + Vertical Flip)
    USE_TTA = True
    TTA_STEPS = 3

    # ==== Debugging ====
    # If True, limits dataset size for quick pipeline testing
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100


# Set seed immediately upon import
seed_everything(Config.SEED)
