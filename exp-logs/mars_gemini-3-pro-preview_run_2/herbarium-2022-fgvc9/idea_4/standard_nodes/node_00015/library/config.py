import os
import torch
import random
import numpy as np


class Config:
    """
    Centralized configuration for the Cascaded Taxonomic Network (CTN).
    """

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORK_DIR = "./working/idea_4"

    # Ensure working directory exists
    os.makedirs(WORK_DIR, exist_ok=True)

    # Input Data Paths
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")
    TRAIN_META_JSON = os.path.join(INPUT_DIR, "train_metadata.json")

    # Metadata CSV Paths (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure submission directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Caching Paths
    TAXONOMY_MAP_PATH = os.path.join(WORK_DIR, "taxonomy_mapping.parquet")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Backbone
    MODEL_NAME = "tf_efficientnet_b4_ns"  # EfficientNet-B4 with Noisy Student weights
    IMG_SIZE = 380  # Native resolution for B4
    BACKBONE_OUT_DIM = 1792  # efficientnet_b4 output channels

    # Classification Heads
    NUM_CLASSES = 15501  # Species count
    EMBEDDING_SIZE = 512  # Dimension for ArcFace embedding

    # ArcFace Parameters
    ARCFACE_SCALE = 30.0
    ARCFACE_MARGIN = 0.50

    # Loss Weights (Cascaded Multi-Task)
    LAMBDA_FAMILY = 1.0
    LAMBDA_GENUS = 1.0

    # ==========================================
    # Training Dynamics
    # ==========================================
    SEED = 42
    BATCH_SIZE = 128
    EPOCHS = 30

    # Optimization
    LR_START = 1e-4
    LR_MIN = 1e-6
    WEIGHT_DECAY = 1e-4  # AdamW default

    # Augmentation (RandAugment)
    RA_N = 2
    RA_M = 9

    # Compute
    NUM_WORKERS = 8  # Optimized for 12 vCPUs
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # Utilities
    # ==========================================
    DEBUG = False  # Toggle for debugging on subsets

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("\n" + "=" * 40)
        print(f"CONFIGURATION: {cls.__name__}")
        print("=" * 40)
        print(f"Device: {cls.DEVICE}")
        print(f"Model: {cls.MODEL_NAME}")
        print(f"Input Size: {cls.IMG_SIZE}x{cls.IMG_SIZE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Learning Rate: {cls.LR_START} -> {cls.LR_MIN} (Cosine)")
        print(f"ArcFace: s={cls.ARCFACE_SCALE}, m={cls.ARCFACE_MARGIN}")
        print(f"Work Dir: {cls.WORK_DIR}")
        print("=" * 40 + "\n")


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
