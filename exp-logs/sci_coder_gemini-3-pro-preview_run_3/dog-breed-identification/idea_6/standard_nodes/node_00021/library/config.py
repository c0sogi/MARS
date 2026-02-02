import os
import torch


class Config:
    """
    Configuration class for the Dog Breed Classification Task.
    Centralizes settings for the Heterogeneous Stratified Ensemble with SWA and Temperature Calibration.
    """

    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use when DEBUG is True

    # ==========================================
    # Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for saving checkpoints, cache, and submissions
    WORKING_DIR = "./working/idea_6"
    OUTPUT_DIR = "./working/idea_6"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # ==========================================
    # Data Configuration
    # ==========================================
    IMG_SIZE = 224
    NUM_CLASSES = 120

    # Geometric Diversity Augmentation Pipeline
    # 1. RandomResizedCrop
    AUG_CROP_SCALE = (0.08, 1.0)
    AUG_CROP_RATIO = (3.0 / 4.0, 4.0 / 3.0)
    # 2. RandomHorizontalFlip
    AUG_FLIP_PROB = 0.5
    # 3. RandAugment
    AUG_RANDAUG_NUM_OPS = 2
    AUG_RANDAUG_MAGNITUDE = 9

    # ==========================================
    # Compute Configuration
    # ==========================================
    NUM_WORKERS = 12
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP = True  # Use Automatic Mixed Precision

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    N_FOLDS = 5
    EPOCHS = 30
    BATCH_SIZE = 64  # Optimized for A100 40GB

    # Optimization
    LR_BACKBONE = 1e-4
    LR_HEAD = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler: Cosine Annealing
    SCHEDULER_T_MAX = 30
    SCHEDULER_MIN_LR = 1e-6

    # Two-Phase Training Strategy
    FREEZE_BACKBONE_EPOCHS = 1

    # ==========================================
    # Model Architecture (Heterogeneous Ensemble)
    # ==========================================
    # Using distinct architectures to maximize feature diversity
    MODEL_ARCHS = [
        # ConvNeXt Small: Excellent texture/local feature extraction
        "convnext_small.fb_in22k_ft_in1k",
        # Swin Transformer Small: Excellent global shape/context extraction
        "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
    ]

    # ==========================================
    # Advanced Techniques
    # ==========================================
    # Stochastic Weight Averaging (SWA)
    USE_SWA = True
    SWA_START_EPOCH = 22  # Activate in the last ~25% of epochs
    SWA_LR = 2e-4

    # Post-hoc Calibration
    USE_TEMP_SCALING = True

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 40)
        print(f"Configuration ({cls.__name__})")
        print("=" * 40)
        for key, value in cls.__dict__.items():
            if not key.startswith("__") and not callable(value):
                print(f"{key}: {value}")
        print("=" * 40)
