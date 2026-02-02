import os
import torch
import random
import numpy as np


class Config:
    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Data Directories
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")
    SEGMENTATIONS_DIR = os.path.join(INPUT_DIR, "segmentations")

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")
    BOUNDING_BOXES_PATH = os.path.join(INPUT_DIR, "train_bounding_boxes.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    # Using 'idea_8' as specified in the prompt requirements for caching
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_8")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # --- Data Processing ---
    # Image resolution: 256x256 is chosen to balance detail with memory usage
    # for a sequence length of 64 on the provided GPU.
    IMAGE_SIZE = (256, 256)

    # Sequence Length: Number of slices sampled per scan
    SEQ_LEN = 64

    # 2.5D Stacking: 3 channels [z-1, z, z+1]
    IN_CHANNELS = 3

    # DICOM Windowing (Bone Window)
    WINDOW_LEVEL = 400
    WINDOW_WIDTH = 1800

    # --- Model Architecture ---
    BACKBONE = "resnet18"
    NUM_CLASSES = 7  # C1-C7

    # --- Training Hyperparameters ---
    SEED = 42
    BATCH_SIZE = 8  # Small batch size acts as regularizer
    NUM_WORKERS = 4  # 12 vCPUs available
    EPOCHS = 10

    # Optimizer
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler
    # T_max set to 1.5 * EPOCHS for decoupled cosine annealing
    T_MAX_MULT = 1.5
    MIN_LR = 1e-6

    # Loss Weights
    # Alpha for the Box-Guided Loss component
    ALPHA_BOX_LOSS = 1.0

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Sets up the environment:
        1. Creates necessary directories.
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.WORKING_DIR, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior where possible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @classmethod
    def get_transforms(cls, mode="train"):
        """
        Returns the appropriate Albumentations transforms.
        Note: Actual implementation of transforms usually requires albumentations import.
        This is a placeholder config method to define parameters if needed.
        """
        pass
