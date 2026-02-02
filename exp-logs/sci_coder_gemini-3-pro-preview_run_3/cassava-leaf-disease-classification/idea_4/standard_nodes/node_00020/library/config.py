import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    NUM_WORKERS = 8  # Optimized for 12 vCPUs
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "swin_large_best.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    IMG_SIZE = 384  # Native resolution for Swin-Large window 12
    NUM_CLASSES = 5

    # Augmentation Settings
    # We prioritize geometric augmentations and mixing
    MIXUP_ALPHA = 0.8
    CUTMIX_ALPHA = 1.0
    MIXUP_PROB = 0.5  # Probability of applying MixUp/CutMix batch-wise

    # Normalization (ImageNet defaults)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # =========================================================================
    # Model Configuration
    # =========================================================================
    # Swin Transformer Large, Patch 4, Window 12, 384px
    # Pretrained on ImageNet-22k ('ms_in22k' tag usually implies this in timm)
    MODEL_NAME = "swin_large_patch4_window12_384.ms_in22k"
    PRETRAINED = True
    DROP_RATE = 0.0
    DROP_PATH_RATE = 0.1  # Stochastic depth

    # =========================================================================
    # Training Configuration
    # =========================================================================
    EPOCHS = 10  # Swin converges relatively fast
    BATCH_SIZE = (
        8  # Adjusted for A100 memory with 384x384 Swin-Large (approx 16GB+ usage)
    )
    ACCUMULATION_STEPS = 2  # Effective batch size = 16

    # Optimizer (AdamW)
    LEARNING_RATE = 2e-5  # Lower LR for fine-tuning large models
    WEIGHT_DECAY = 0.05
    EPS = 1e-8

    # Scheduler (Cosine Annealing)
    MIN_LR = 1e-6
    T_MAX = EPOCHS  # For CosineAnnealingLR

    # Loss
    LABEL_SMOOTHING = 0.1

    # Early Stopping
    PATIENCE = 3  # Stop if validation accuracy doesn't improve for 3 epochs

    # =========================================================================
    # Inference Configuration
    # =========================================================================
    USE_TTA = True  # Test Time Augmentation (Horizontal Flip)
    TTA_STEPS = 2  # Original + Flip

    @classmethod
    def setup_directories(cls):
        """Ensures necessary directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
