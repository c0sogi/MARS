import os
import torch


class Config:
    """
    Central configuration for the Hybrid CNN-Transformer Ensemble (Idea 7).
    Stores hyperparameters, file paths, and system settings.
    """

    # -------------------------------------------------------------------------
    # System & Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42
    # Use GPU if available
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Adjust workers based on available vCPUs (12 available)
    NUM_WORKERS = 8

    # -------------------------------------------------------------------------
    # File Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for checkpoints and cached data
    WORKING_DIR = "./working/idea_7"

    # Submission directory
    SUBMISSION_DIR = "./submission"

    # Ensure writable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Final Submission Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    # Image Resolution: 256x256 (Proven improvement over 224)
    IMG_SIZE = 256

    # Interpolation: Bicubic (Critical for Swin/ConvNeXt)
    INTERPOLATION = "bicubic"

    # Debugging Flags
    DEBUG = False
    DEBUG_SUBSET_SIZE = 500  # Size of subset when DEBUG is True

    # -------------------------------------------------------------------------
    # Model Configuration
    # -------------------------------------------------------------------------
    # Hybrid Heterogeneous Ensemble:
    # 1. ResNet50 (Standard CNN): Robust, texture-biased.
    # 2. ConvNeXt-Small (Modern CNN): Large kernel, semantic context.
    # 3. Swin-Tiny (Transformer): Global dependency modeling via self-attention.
    # Using 'timm' library naming conventions.
    MODEL_BACKBONES = [
        "resnet50.a1_in1k",  # ResNet50 with modern 'a1' recipe
        "convnext_small.fb_in1k",  # ConvNeXt Small
        "swin_tiny_patch4_window7_224.ms_in1k",  # Swin Tiny (Microsoft weights)
    ]

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    EPOCHS = 6
    BATCH_SIZE = 32  # Adjusted for 40GB GPU memory with multiple models

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01  # Standard for AdamW
    OPTIMIZER_NAME = "AdamW"

    # Scheduler
    SCHEDULER_NAME = "CosineAnnealingLR"
    MIN_LR = 1e-6

    # Loss Function
    # Strictly NO label smoothing to avoid loss floor
    LABEL_SMOOTHING = 0.0

    # -------------------------------------------------------------------------
    # Augmentation
    # -------------------------------------------------------------------------
    # Context-Preservation
    CROP_SCALE = (0.8, 1.0)

    # Photometric Noise (Intensity >= 0.2)
    COLOR_JITTER_BRIGHTNESS = 0.2
    COLOR_JITTER_CONTRAST = 0.2
    COLOR_JITTER_SATURATION = 0.2
    COLOR_JITTER_HUE = 0.0

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    # Test Time Augmentation: Horizontal Flip
    USE_TTA = True
