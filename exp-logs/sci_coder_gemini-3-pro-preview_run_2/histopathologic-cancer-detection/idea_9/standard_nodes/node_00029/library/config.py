import os
import torch


class Config:
    """
    Central configuration for the Memory-Resident Heterogeneous Ensemble pipeline.
    Defines hyperparameters, paths, and system settings.
    """

    # --- Project Information ---
    PROJECT_NAME = "idea_9"
    SEED = 42
    DEBUG = False  # Set to True for fast debugging runs

    # --- Directories ---
    # Read-only input directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

    # Working directories (Write access allowed)
    WORK_DIR = os.path.join("./working", PROJECT_NAME)
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORK_DIR, "submission")

    # Ensure working directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --- Data Configuration ---
    # Dimensions
    ORIG_IMAGE_SIZE = 96  # Original patch size
    CROP_SIZE = 64  # Input size to the network (Contextual Crop)
    ROI_SIZE = 32  # Region of Interest (Center) - Implicitly covered by label logic

    # Normalization (Calculated from EDA)
    MEAN = [0.7035, 0.5476, 0.6975]
    STD = [0.2388, 0.2821, 0.2159]

    # Loading Strategy
    LOAD_TO_RAM = True  # Load all images into memory to eliminate I/O bottlenecks
    NUM_CLASSES = 1

    # --- Model Configuration ---
    # Heterogeneous Ensemble Backbones
    # 1. ConvNeXt: Isotropic/Transformer-style blocks
    # Removed EfficientNet to avoid ensemble drag-down (Cite solution_lesson_node_00027)
    MODEL_BACKBONES = ["convnext_tiny"]

    # Exponential Moving Average
    USE_EMA = True
    EMA_DECAY = (
        0.999  # Tuned for 15 epochs (~16k steps) (Cite solution_lesson_node_00024)
    )

    # --- Training Hyperparameters ---
    NUM_FOLDS = 5
    EPOCHS = 15
    BATCH_SIZE = 128

    # Optimization
    LEARNING_RATE = 2e-4  # Conservative LR for fine-tuning
    WEIGHT_DECAY = 0.05  # High weight decay for regularization
    MIN_LR = 1e-6  # Cosine annealing minimum

    # Regularization
    MIXUP_ALPHA = 0.2

    # Early Stopping
    PATIENCE = 5  # Stop if validation metric doesn't improve

    # --- Inference Configuration ---
    TTA_VIEWS = 8  # 8-view Dihedral TTA (Rotations + Flips)

    # --- Hardware ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Matches available vCPUs
    PIN_MEMORY = True

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 30)
        print(f"Configuration: {cls.PROJECT_NAME}")
        print("=" * 30)
        print(f"Device: {cls.DEVICE}")
        print(f"Models: {cls.MODEL_BACKBONES}")
        print(f"Image Size: {cls.ORIG_IMAGE_SIZE} -> {cls.CROP_SIZE} (Crop)")
        print(f"Load to RAM: {cls.LOAD_TO_RAM}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Mixup Alpha: {cls.MIXUP_ALPHA}")
        print(f"EMA Decay: {cls.EMA_DECAY}")
        print(f"TTA Views: {cls.TTA_VIEWS}")
        print("=" * 30)
