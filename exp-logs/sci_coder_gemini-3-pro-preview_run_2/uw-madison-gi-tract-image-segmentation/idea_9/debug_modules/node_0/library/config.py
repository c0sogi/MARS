import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration module for the Stomach and Intestines Segmentation project.
    Implements settings for Idea 9: 2.5D Bilateral Segmentation Network (BiSeNet).
    """

    # =========================================================================
    # Path Configuration
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_9"
    SUBMISSION_DIR = "./submission"

    # Metadata Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    CACHE_PATH = os.path.join(WORKING_DIR, "data_cache.parquet")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    # Idea 9: 2.5D inputs (Slice i-1, Slice i, Slice i+1) -> 3 Channels
    IN_CHANNELS = 3
    # Resize to 256x256 for efficiency as per Idea 9
    IMG_SIZE = (256, 256)

    # Classes: large_bowel, small_bowel, stomach
    NUM_CLASSES = 3
    CLASS_LABELS = ["large_bowel", "small_bowel", "stomach"]

    # Sampling Strategy: Train on all positives + 50% subsample of negatives
    NEGATIVE_SAMPLING_RATIO = 0.5

    # =========================================================================
    # Model Configuration
    # =========================================================================
    MODEL_NAME = "BiSeNet_2.5D"
    BACKBONE = "mobilenet_v2"  # Lightweight backbone for Context Path

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Loss Weights for Idea 9
    # L_total = L_primary + AUX_LOSS_WEIGHT * L_aux
    AUX_LOSS_WEIGHT = 0.1

    # Metric Weights (Used for evaluation/monitoring, not loss)
    METRIC_DICE_WEIGHT = 0.4
    METRIC_HAUSDORFF_WEIGHT = 0.6

    # =========================================================================
    # Hardware & Compute
    # =========================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 12 vCPUs available, using 10 for workers to leave overhead for main process
    NUM_WORKERS = 10

    # =========================================================================
    # Debugging & Flexibility
    # =========================================================================
    DEBUG = False
    # If DEBUG is True, limit dataset to this size
    DEBUG_SAMPLE_SIZE = 500

    @classmethod
    def setup(cls, seed=None):
        """
        Initialize the environment:
        1. Create necessary directories.
        2. Set random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set Seed
        if seed is None:
            seed = cls.SEED

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
