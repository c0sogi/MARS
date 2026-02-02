import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for HuBMAP Kidney Segmentation.
    Centralizes hyperparameters, file paths, and model settings.
    """

    # --- General Settings ---
    SEED = 42
    DEBUG = False  # Set to True to run on a subset of data for debugging
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 8  # Optimized for the available 12 vCPUs

    # --- Directories ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for artifacts (checkpoints, cache, etc.)
    # Using 'idea_2' to denote the updated strategy
    WORKING_DIR = "./working/idea_2"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Ensure writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --- File Paths ---
    # Metadata files (Pre-generated in ./metadata)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output paths
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Data Configuration ---
    TILE_SIZE = 1024
    # Stride for inference (sliding window). Set equal to TILE_SIZE for non-overlapping.
    INFERENCE_STRIDE = 1024

    # Normalization (ImageNet stats)
    PIXEL_MEAN = [0.485, 0.456, 0.406]
    PIXEL_STD = [0.229, 0.224, 0.225]

    # Sampling Strategy
    # Probability of selecting a tile that intersects with Cortex (Anatomical Guidance)
    CORTEX_SAMPLING_PROB = 0.8

    # --- Model Architecture ---
    # U-Net++ with EfficientNet-B5 backbone
    ARCH = "UnetPlusPlus"
    ENCODER = "efficientnet-b5"
    ENCODER_WEIGHTS = "imagenet"
    IN_CHANNELS = 3
    CLASSES = 1
    ACTIVATION = None  # Output logits for BCEWithLogitsLoss

    # --- Training Hyperparameters ---
    # EfficientNet-B5 is VRAM intensive. Reduced to 2 for 16GB VRAM at 1024x1024.
    BATCH_SIZE = 2
    EPOCHS = 25
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    MIN_LR = 1e-6

    # Loss Weights
    DICE_WEIGHT = 0.5
    BCE_WEIGHT = 0.5

    # --- Debugging ---
    # If DEBUG is True, limit data to this many samples
    DEBUG_SAMPLES = 100

    @staticmethod
    def setup_reproducibility(seed=42):
        """
        Sets the random seed for python, numpy, and torch to ensure reproducibility.
        """
        Config.SEED = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"Random seed set to {seed}")

    @staticmethod
    def get_config_dict():
        """
        Returns a dictionary of configuration parameters relevant for data hashing/caching.
        This ensures that if tile size or preprocessing changes, the cache is invalidated.
        """
        return {
            "tile_size": Config.TILE_SIZE,
            "pixel_mean": Config.PIXEL_MEAN,
            "pixel_std": Config.PIXEL_STD,
            "seed": Config.SEED,
            "cortex_prob": Config.CORTEX_SAMPLING_PROB,
        }
