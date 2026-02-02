import os
import torch
import random
import numpy as np


class Config:
    # --- Hyperparameters ---
    seed = 42
    image_size = 224
    batch_size = 128  # A100 has 40GB, 128 is safe for EfficientNetV2-S
    learning_rate = 1e-3
    epochs = 10
    model_name = "tf_efficientnetv2_s"
    num_classes = 3474
    num_workers = 12  # Using available vCPUs

    # Debug mode: set to True to run on a small subset for testing
    debug = False

    # Caching
    load_cached_data = True

    # --- Hardware ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files (pre-generated)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Raw labels description
    LABELS_PATH = os.path.join(INPUT_DIR, "labels.csv")

    # Working directory for artifacts
    WORKING_DIR = "./working"
    IDEA_DIR = os.path.join(WORKING_DIR, "idea_1")

    # Model checkpoints
    MODEL_PATH = os.path.join(IDEA_DIR, "best_model.pth")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    @classmethod
    def setup(cls):
        """
        Creates necessary directories and sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.IDEA_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set random seeds
        random.seed(cls.seed)
        np.random.seed(cls.seed)
        torch.manual_seed(cls.seed)
        torch.cuda.manual_seed_all(cls.seed)

        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
