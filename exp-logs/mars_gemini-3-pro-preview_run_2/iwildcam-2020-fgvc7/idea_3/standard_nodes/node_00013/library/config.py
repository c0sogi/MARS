import os
import random
import numpy as np
import torch


class Config:
    # --------------------------------------------------------------------------
    # Paths & Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Auxiliary Data
    MEGADETECTOR_FILE = os.path.join(
        INPUT_DIR, "iwildcam2020_megadetector_results.json"
    )
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    CACHE_FILE = os.path.join(WORKING_DIR, "processed_data_cache.parquet")
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model_effnetv2m.pth")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------
    MODEL_NAME = "tf_efficientnetv2_m.in21k"
    NUM_CLASSES = 676  # Categories 0 to 675

    # Input specs
    IMG_SIZE = 448

    # Training Loop
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2  # Standard for AdamW
    LABEL_SMOOTHING = 0.1
    MAX_GRAD_NORM = 10.0

    # Regularization (Mixup/Cutmix)
    MIXUP_ALPHA = 0.8
    CUTMIX_ALPHA = 1.0
    MIXUP_PROB = 0.5  # Probability of applying mixup/cutmix

    # --------------------------------------------------------------------------
    # Compute & Environment
    # --------------------------------------------------------------------------
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------------------------
    # Debugging & Development
    # --------------------------------------------------------------------------
    # Set DEBUG to True to run on a small subset of data for quick testing
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 2000

    @classmethod
    def setup(cls):
        """
        Creates necessary directories and sets random seeds.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        cls.seed_everything(cls.SEED)

        # Print config status
        print(f"Config Setup Complete.")
        print(f"  Device: {cls.DEVICE}")
        print(f"  Model: {cls.MODEL_NAME}")
        print(f"  Image Size: {cls.IMG_SIZE}x{cls.IMG_SIZE}")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print(f"  Epochs: {cls.EPOCHS}")

    @staticmethod
    def seed_everything(seed: int):
        """
        Sets seeds for random, numpy, and torch to ensure reproducibility.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
