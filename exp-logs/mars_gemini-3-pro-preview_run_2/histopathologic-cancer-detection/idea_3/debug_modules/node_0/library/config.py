import os
import torch


class Config:
    """
    Central configuration for the Pathology Tumor Detection task.
    Implements parameters for ConvNeXt-Small, Mixup, EMA, and TTA strategies.
    """

    # --- Directory Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific experiment (Idea 3)
    WORKING_DIR = "./working/idea_3"

    # Sub-directories for organization
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Metadata File Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Output File Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    LOG_FILE = os.path.join(WORKING_DIR, "training.log")

    # --- Hardware & Reproducibility ---
    SEED = 42
    # Detect device (A100 is expected)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Utilize all 12 vCPUs for data loading
    NUM_WORKERS = 12

    # --- Data Configuration ---
    # Full patch size from disk
    FULL_IMAGE_SIZE = 96
    # Input size to the model (Contextual ROI)
    CROP_SIZE = 64
    # Center region size (target definition, for reference)
    ROI_SIZE = 32

    # Normalization Statistics (Calculated from EDA)
    # R, G, B channels
    DATASET_MEAN = [0.7035, 0.5476, 0.6975]
    DATASET_STD = [0.2388, 0.2821, 0.2159]

    # Debugging
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 2000  # Subset size for rapid debugging

    # --- Model Configuration ---
    # Backbone: ConvNeXt-Small (Pretrained on ImageNet-1k)
    MODEL_NAME = "convnext_small.fb_in1k"
    PRETRAINED = True
    NUM_CLASSES = 1

    # Dropout & Stochastic Depth
    DROP_RATE = 0.0
    DROP_PATH_RATE = 0.2  # Regularization for deeper networks

    # --- Training Configuration ---
    NUM_EPOCHS = 20
    # Batch size optimized for A100 (40GB VRAM)
    BATCH_SIZE = 256

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.05

    # Scheduler (Cosine Annealing)
    ETA_MIN = 1e-6

    # Regularization: Mixup
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2

    # Model Exponential Moving Average (EMA)
    USE_EMA = True
    EMA_DECAY = 0.9999

    # Early Stopping
    PATIENCE = 5

    # --- Inference Configuration ---
    # Test Time Augmentation (8-view: 4 rotations * 2 flips)
    USE_TTA = True
    TTA_VIEWS = 8

    @classmethod
    def setup(cls):
        """
        Initialize the workspace by creating necessary directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        print(f"Configuration Initialized:")
        print(f"  - Device: {cls.DEVICE}")
        print(f"  - Model: {cls.MODEL_NAME}")
        print(
            f"  - Input: {cls.CROP_SIZE}x{cls.CROP_SIZE} (cropped from {cls.FULL_IMAGE_SIZE})"
        )
        print(f"  - Batch Size: {cls.BATCH_SIZE}")
        print(f"  - Mixup: {cls.USE_MIXUP} (Alpha={cls.MIXUP_ALPHA})")
        print(f"  - EMA: {cls.USE_EMA}")
