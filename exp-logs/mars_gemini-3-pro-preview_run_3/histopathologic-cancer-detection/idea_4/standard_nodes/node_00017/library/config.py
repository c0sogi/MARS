import os
import torch
import numpy as np
import random


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORK_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    IMAGE_SIZE = 96  # Original image dimensions
    CROP_SIZE = 64  # Input size to the model (Center Crop)
    ROI_SIZE = 32  # Target region size (center 32x32)
    NUM_CLASSES = 1  # Binary classification

    # ==========================================
    # Model Configuration
    # ==========================================
    # Heterogeneous Ensemble: ConvNeXt-Tiny and EfficientNetV2-Small
    MODEL_ARCHS = [
        "convnext_tiny.fb_in22k_ft_in1k",
        "tf_efficientnetv2_s.in21k_ft_in1k",
    ]

    # ==========================================
    # Training Configuration
    # ==========================================
    SEED = 42
    NUM_FOLDS = 5  # 5-Fold Cross-Validation
    NUM_EPOCHS = 30  # Extended training for convergence
    BATCH_SIZE = 512  # Maximized for A100
    LEARNING_RATE = 1e-4  # Moderate LR for fine-tuning
    WEIGHT_DECAY = 1e-4  # Regularization
    NUM_WORKERS = 12  # Available vCPUs

    # ==========================================
    # Inference Configuration
    # ==========================================
    TTA_VIEWS = 4  # Original + HFlip + VFlip + Combined

    # ==========================================
    # Compute Configuration
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def setup():
        """
        Initializes directories and sets random seeds for reproducibility.
        """
        # Create necessary directories
        os.makedirs(Config.WORK_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Set Random Seeds
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.SEED)
            # Ensure deterministic behavior where possible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
