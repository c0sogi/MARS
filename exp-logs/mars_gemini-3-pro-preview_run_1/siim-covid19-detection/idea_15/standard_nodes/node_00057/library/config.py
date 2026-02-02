import os
import torch
import random
import numpy as np


class Config:
    """
    Centralized configuration for the DropBlock-Regularized ResNet34 Multi-Task U-Net.
    """

    # ==========================
    # Directories & Paths
    # ==========================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_15"
    SUBMISSION_DIR = "./submission"

    # Metadata Files (Generated in previous step)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # ==========================
    # Data Configuration
    # ==========================
    IMG_SIZE = 512
    NUM_CLASSES = 4
    CLASS_LABELS = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]

    # ==========================
    # Model Architecture
    # ==========================
    BACKBONE = "resnet34"
    PRETRAINED = True

    # DropBlock Regularization
    DROPBLOCK_PROB = 0.1  # Probability of dropping a block
    DROPBLOCK_BLOCK_SIZE = 7  # Size of the block to drop

    # ==========================
    # Training Hyperparameters
    # ==========================
    SEED = 42
    EPOCHS = 20
    BATCH_SIZE = 32  # A100 40GB allows for larger batch sizes

    # Optimizer settings
    # Linear Scaling Rule: Scale LR based on batch size relative to a base reference.
    # Reference: LR 1e-4 for Batch Size 16.
    REF_BATCH_SIZE = 16
    REF_LR = 1e-4
    LEARNING_RATE = REF_LR * (BATCH_SIZE / REF_BATCH_SIZE)

    WEIGHT_DECAY = 1e-2
    MIN_LR = 1e-6

    # Loss Function Weights
    # Prioritize segmentation (10.0) over classification (1.0) to force spatial feature learning
    CLS_LOSS_WEIGHT = 1.0
    SEG_LOSS_WEIGHT = 10.0

    # ==========================
    # Hardware & System
    # ==========================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Matches available vCPUs

    # ==========================
    # Debugging & Development
    # ==========================
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100  # Number of images to use when DEBUG is True

    @classmethod
    def setup(cls):
        """
        Prepares the environment: creates directories and sets random seeds.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set reproducible seeds
        cls.set_seed(cls.SEED)

        # Print configuration summary
        cls.print_config()

    @staticmethod
    def set_seed(seed):
        """Sets seeds for all random number generators."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print(f"\n{'='*20} CONFIGURATION {'='*20}")
        print(f"Device: {cls.DEVICE}")
        print(f"Model Backbone: {cls.BACKBONE}")
        print(f"Image Size: {cls.IMG_SIZE}x{cls.IMG_SIZE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(
            f"Learning Rate: {cls.LEARNING_RATE:.2e} (Scaled from base {cls.REF_LR} @ BS={cls.REF_BATCH_SIZE})"
        )
        print(f"Epochs: {cls.EPOCHS}")
        print(
            f"Loss Weights -> Class: {cls.CLS_LOSS_WEIGHT}, Seg: {cls.SEG_LOSS_WEIGHT}"
        )
        print(f"Debug Mode: {cls.DEBUG}")
        print(f"{'='*55}\n")
