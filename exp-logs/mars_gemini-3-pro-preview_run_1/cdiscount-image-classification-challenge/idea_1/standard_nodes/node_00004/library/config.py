import os
import torch
import random
import numpy as np


class Config:
    # ==========================================
    # PATHS
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")

    # Raw Data Files
    TRAIN_BSON = os.path.join(INPUT_DIR, "train.bson")
    TEST_BSON = os.path.join(INPUT_DIR, "test.bson")
    CATEGORY_NAMES = os.path.join(INPUT_DIR, "category_names.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Pre-generated)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_CHECKPOINT = os.path.join(WORKING_DIR, "resnet18_best.pth")

    # ==========================================
    # DATA PARAMETERS
    # ==========================================
    IMG_SIZE = 180  # Native size of images in BSON
    NUM_CLASSES = 5270  # Total number of product categories
    MEAN = [0.485, 0.456, 0.406]  # ImageNet Mean
    STD = [0.229, 0.224, 0.225]  # ImageNet Std

    # ==========================================
    # MODEL HYPERPARAMETERS
    # ==========================================
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    EMBEDDING_SIZE = 512  # Output feature size of ResNet18 before FC

    # ==========================================
    # TRAINING HYPERPARAMETERS
    # ==========================================
    SEED = 42
    BATCH_SIZE = 512  # Tuned for A100 40GB (considering variable img count per product)
    NUM_EPOCHS = 2  # Sufficient for OneCycleLR convergence
    LEARNING_RATE = 1e-3  # Max LR for OneCycleLR
    WEIGHT_DECAY = 1e-4

    # Early Stopping
    PATIENCE = 3
    MIN_DELTA = 0.0001

    # ==========================================
    # HARDWARE SETTINGS
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 8  # Optimal for 12 vCPUs
    PIN_MEMORY = True

    @classmethod
    def setup_directories(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)


def seed_everything(seed=Config.SEED):
    """Sets the random seed for reproducibility across all libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Initialize directories immediately when config is imported
Config.setup_directories()
