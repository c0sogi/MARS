import os
import torch


class Config:
    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORK_DIR = "./working/idea_5"
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    UNICODE_MAP_PATH = os.path.join(INPUT_DIR, "unicode_translation.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Image Directories
    TRAIN_IMGS_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMGS_DIR = os.path.join(INPUT_DIR, "test_images")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    BEST_MODEL_PATH = os.path.join(WORK_DIR, "best_model.pth")
    LOG_PATH = os.path.join(WORK_DIR, "training_log.csv")

    # =========================================================================
    # Reproducibility
    # =========================================================================
    SEED = 42

    # =========================================================================
    # Data Configuration
    # =========================================================================
    # Set to a small integer (e.g., 100) for debugging, or None for full dataset
    DEBUG_SAMPLE_SIZE = None

    # Input Dimensions
    IMG_SIZE = (1024, 1024)
    IN_CHANNELS = 3

    # Normalization (ImageNet Statistics)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # Augmentation (Geometric Only - No Photometric Distortions)
    SCALE_RANGE = (0.5, 1.5)
    ROTATION_DEG = 15
    SHIFT_LIMIT = 0.1

    # =========================================================================
    # Model Configuration
    # =========================================================================
    # Backbone: Swin Transformer Base
    BACKBONE = "swin_base_patch4_window7_224"

    # Classes: Based on unicode_translation.csv (4782 lines)
    NUM_CLASSES = 4782

    # Output Stride for the CenterNet Head (typically 4)
    OUTPUT_STRIDE = 4

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_EPOCHS = 35
    BATCH_SIZE = 4  # Conservative for Swin-B @ 1024x1024 on 40GB VRAM
    NUM_WORKERS = 4

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.05
    CLIP_GRAD = 1.0

    # Scheduler (Cosine Annealing)
    WARMUP_EPOCHS = 3
    MIN_LR = 1e-6

    # Loss Weights
    HM_WEIGHT = 1.0  # Heatmap Loss (Focal)
    WH_WEIGHT = 0.1  # Size Regression Loss (L1)
    OFF_WEIGHT = 1.0  # Offset Regression Loss (L1)

    # =========================================================================
    # Inference Configuration
    # =========================================================================
    CONF_THRESHOLD = 0.1
    MAX_DETECTIONS = 1200

    @classmethod
    def setup(cls):
        """Creates necessary working and submission directories."""
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
