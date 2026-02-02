import os
import torch
import random
import numpy as np


class Config:
    # ==============================
    # General & Paths
    # ==============================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging

    # Input Paths
    INPUT_DIR = "./input"
    TRAIN_METADATA_PATH = os.path.join("./metadata", "train.csv")
    VALID_METADATA_PATH = os.path.join("./metadata", "validation.csv")
    TEST_METADATA_PATH = os.path.join("./metadata", "test.csv")

    # Output/Working Paths
    # Using 'idea_6' as implied by the progression of ideas in the prompt context
    WORKING_DIR = "./working/idea_6"
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = "submission.csv"

    # ==============================
    # Data Preprocessing (Fine-Grained Stratified)
    # ==============================
    # Image Dimensions
    TILE_SIZE = 512
    STRIDE = 512

    # Z-Axis Slicing Strategy
    # We take 6 slabs of 4 slices each, starting from index 20.
    # Total Z-depth covered: 6 * 4 = 24 slices (Indices 20 to 43)
    Z_START = 20
    SLAB_COUNT = 6
    SLAB_DEPTH = 4

    # Input channels = number of slabs (since we do 1 MIP per slab)
    IN_CHANNELS = SLAB_COUNT  # 6

    # Normalization
    # We assume data is uint16, will normalize to [0, 1]
    PIXEL_MIN = 0.0
    PIXEL_MAX = 65535.0

    # ==============================
    # Model Architecture
    # ==============================
    # SegFormer MiT-B2
    BACKBONE = "nvidia/mit-b2"
    PRETRAINED = True
    NUM_CLASSES = 1  # Binary segmentation (Ink vs No Ink)

    # ==============================
    # Training Hyperparameters
    # ==============================
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    EPOCHS = 15

    # Optimization
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 2
    SCHEDULER_MIN_LR = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5

    # Loss Weights (BCE + Dice)
    BCE_WEIGHT = 0.5
    DICE_WEIGHT = 0.5

    # Metric Threshold
    # We only save/submit if we beat the previous best F0.5 score
    BASELINE_SCORE_THRESHOLD = 0.551

    # ==============================
    # Hardware
    # ==============================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_reproducibility(seed=Config.SEED):
    """
    Sets random seeds for python, numpy, and torch to ensure reproducibility.
    Creates the working directory.
    """
    # Create working directory
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set env var for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Reproducibility setup complete. Seed: {seed}")
    print(f"Working directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")
