import os
import torch


class Config:
    """
    Configuration class for the Sharpness-Aware SWA-ResNet Ensemble solution.
    Defines file paths, data statistics, model hyperparameters, and training settings.
    """

    # =========================================================================
    # Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORK_DIR = "./working/idea_23"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Data Statistics (Global Min-Max)
    # Derived from training set analysis
    # =========================================================================
    BAND1_MIN = -45.5944
    BAND1_MAX = 32.1806
    BAND2_MIN = -45.6555
    BAND2_MAX = 17.8628

    # =========================================================================
    # Data Processing
    # =========================================================================
    IMG_SIZE = 224  # Upsampled size
    INTERPOLATION = "bicubic"  # For resizing

    # =========================================================================
    # Model Architecture
    # =========================================================================
    MODEL_ARCH = "resnet18"
    PRETRAINED = True
    DROPOUT_RATE = 0.5
    NUM_CLASSES = 1
    USE_LATE_FUSION = True
    FUSION_DIM = 512 + 1  # ResNet18 GAP (512) + Inc Angle (1)

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 64
    NUM_WORKERS = 2  # Adjusted for vCPU availability
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization (SAM + AdamW)
    LR = 2e-4
    WEIGHT_DECAY = 0.01
    SAM_RHO = 0.05

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 10

    # Loss Function
    LABEL_SMOOTHING = 0.05

    # =========================================================================
    # Training Protocol
    # =========================================================================
    # Phase 1: Calibration (Global Epoch Selection)
    N_FOLDS = 5
    MAX_EPOCHS_PHASE1 = 50
    EARLY_STOPPING_PATIENCE = 15

    # Phase 2: Production (Full-Fit SWA)
    # Total Epochs = E_conv (from Phase 1) + SWA_DURATION
    SWA_DURATION = 12
    SWA_LR = 2e-4  # Constant LR for SWA phase (usually same as base or slightly lower)

    # =========================================================================
    # Augmentation
    # =========================================================================
    ROTATION_LIMIT = 20  # Degrees (+/-)
    HORIZONTAL_FLIP_PROB = 0.5
    VERTICAL_FLIP_PROB = 0.5
