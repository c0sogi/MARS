import os
import random
import numpy as np
import torch
import pandas as pd


class Config:
    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific cache directory for this idea
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_32")

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Raw Data
    DEPTHS_CSV_PATH = os.path.join(INPUT_DIR, "depths.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Checkpoint Paths
    TEACHER_CHECKPOINT_DIR = os.path.join(CACHE_DIR, "teacher_checkpoints")
    STUDENT_CHECKPOINT_DIR = os.path.join(CACHE_DIR, "student_checkpoints")

    # =========================================================================
    # Data Parameters
    # =========================================================================
    ORIG_SIZE = 101
    IMG_SIZE = 128  # Pad to 128x128 for U-Net divisibility
    IN_CHANNELS = 1  # Seismic images are grayscale

    # Normalization (Standard ImageNet)
    # Note: Even for 1-channel, we often use these if using pretrained weights,
    # or the code handles conversion.
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 64  # A100 allows larger batches
    NUM_WORKERS = 4

    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Stage 1: Specialist Teacher Ensemble
    STAGE1_EPOCHS = 50
    STAGE1_FOLDS = 5
    # Gating: Discard models with mAP < 0.75
    STAGE1_GATING_THRESHOLD = 0.75

    # Stage 2: Marginalization (Inference params)
    # Scan depths at these sigma deviations
    DEPTH_SCAN_SIGMAS = [-1.5, -0.75, 0, 0.75, 1.5]

    # Stage 3: Generalist Student
    STAGE3_EPOCHS = 50

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10

    # Debugging / Development
    # Set to a small integer (e.g., 100) to limit dataset size for quick tests
    MAX_TRAIN_SAMPLES = None
    MAX_VAL_SAMPLES = None

    # =========================================================================
    # Augmentation Parameters
    # =========================================================================
    AUG_PROB = 0.2

    # Elastic Transform
    AUG_ELASTIC_ALPHA = 120
    AUG_ELASTIC_SIGMA = 6

    # =========================================================================
    # Model Architecture
    # =========================================================================
    BACKBONE = "resnet34"
    PRETRAINED = True
    # Wide-LinkNet decoder channels
    DECODER_CHANNELS = [256, 128, 64, 32, 16]

    # =========================================================================
    # Compute
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Sets up the environment: creates directories and sets random seeds.
        """
        # Create directories
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        os.makedirs(cls.TEACHER_CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.STUDENT_CHECKPOINT_DIR, exist_ok=True)

        # Set seeds for reproducibility
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_depth_stats(load_cached_data=True):
    """
    Calculates or loads mean and std of depth values.

    Args:
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        dict: {'mean': float, 'std': float}
    """
    cache_path = os.path.join(Config.CACHE_DIR, "depth_stats.npy")

    # Ensure directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            stats = np.load(cache_path, allow_pickle=True).item()
            return stats
        except Exception:
            # If load fails, fall through to recompute
            pass

    # 2. Compute from scratch
    df = pd.read_csv(Config.DEPTHS_CSV_PATH)
    depths = df["z"].values.astype(np.float32)

    stats = {"mean": float(np.mean(depths)), "std": float(np.std(depths))}

    # 3. Save to cache
    np.save(cache_path, stats)

    return stats
