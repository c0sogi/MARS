import os
import random
import numpy as np
import torch


class Config:
    """
    Configuration for Stratified ResNet-18 Ensemble with Mixup and SWA.
    """

    # --------------------------------------------------------------------------
    # General & Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for pipeline verification
    DEBUG_SUBSET_SIZE = 100

    # --------------------------------------------------------------------------
    # Compute Environment
    # --------------------------------------------------------------------------
    # 12 vCPUs available
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------------------------
    # File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_8"
    SUBMISSION_DIR = "./submission"

    # Raw Data
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Preprocessing
    # --------------------------------------------------------------------------
    IMG_SIZE = 224  # Upsample to 224x224 for ResNet-18
    IN_CHANNELS = 3  # Band 1 (HH), Band 2 (HV), Mean

    # --------------------------------------------------------------------------
    # Model Architecture
    # --------------------------------------------------------------------------
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    DROPOUT_RATE = 0.5
    NUM_CLASSES = 1  # Binary classification (Ship vs Iceberg)

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    N_FOLDS = 1
    BATCH_SIZE = 32
    NUM_EPOCHS = 35
    LEARNING_RATE = 1e-4  # Initial learning rate for AdamW
    WEIGHT_DECAY = 0.01  # Weight decay for regularization
    PATIENCE = 12  # Early stopping patience

    # --------------------------------------------------------------------------
    # Regularization & Optimization Strategies
    # --------------------------------------------------------------------------
    # Label Smoothing (Cite solution_lesson_node_00005)
    LABEL_SMOOTHING = 0.05

    # Mixup Regularization (Disabled - Cite solution_lesson_node_00030)
    USE_MIXUP = False
    MIXUP_ALPHA = 0.4

    # Stochastic Weight Averaging (Disabled - Cite solution_lesson_node_00030)
    USE_SWA = False
    SWA_START_EPOCH = 25
    SWA_LR = 1e-5

    @classmethod
    def setup(cls):
        """
        Prepare the environment:
        1. Create necessary working and submission directories.
        2. Set random seeds for reproducibility across Python, NumPy, and PyTorch.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
