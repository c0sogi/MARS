import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for Vesuvius Ink Detection.
    Implements settings for Deep 2.5D ResNet34 U-Net with Stratified Depth Projection.
    """

    # --- General ---
    SEED = 42
    DEBUG = False  # Set to True to run on a smaller subset for debugging
    EXP_NAME = "idea_2"

    # --- Compute ---
    # 12 vCPUs available
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Output directories
    WORKING_DIR = os.path.join("./working", EXP_NAME)
    SUBMISSION_PATH = "./submission/submission.csv"

    # Checkpoint paths
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # --- Data / Volume Parameters ---
    TILE_SIZE = 512
    STRIDE = 512  # Stride for tiling

    # Stratified Depth Projection Settings
    # We use the central Z-slices where ink is most likely to be located.
    # Range 22-42 provides 20 slices.
    Z_START = 22
    Z_END = 42

    # We split the Z-range into 3 contiguous sub-volumes to create a 3-channel image.
    # This allows the model to learn depth-dependent features.
    NUM_SUB_VOLUMES = 3
    IN_CHANNELS = 3  # Resulting channels after projection (one per sub-volume)

    # --- Model Architecture ---
    ENCODER_NAME = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    DECODER_CHANNELS = (256, 128, 64, 32, 16)

    # --- Training Hyperparameters ---
    BATCH_SIZE = 16  # Adjusted for A100 GPU
    LEARNING_RATE = 1e-4
    EPOCHS = 15

    # Optimizer & Scheduler
    WEIGHT_DECAY = 1e-6
    SCHEDULER_PATIENCE = 3
    SCHEDULER_FACTOR = 0.5

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5

    # --- Inference ---
    THRESHOLD = 0.5
    USE_TTA = True  # Test Time Augmentation (Horizontal/Vertical Flips)

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        # Ensure the directory for the submission file exists
        submission_dir = os.path.dirname(cls.SUBMISSION_PATH)
        if submission_dir:
            os.makedirs(submission_dir, exist_ok=True)

    @staticmethod
    def set_seed(seed=42):
        """
        Sets fixed random seeds for reproducibility.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
