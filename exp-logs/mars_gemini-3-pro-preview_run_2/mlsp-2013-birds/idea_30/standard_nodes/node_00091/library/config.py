import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    PROJECT_NAME = "bird_species_classification_idea_30"
    SEED = 42
    DEBUG = False  # Set to True for fast debugging runs with subset of data

    # =========================================================================
    # Directory Paths
    # =========================================================================
    # Base Input Directories
    INPUT_DIR = "./input"
    ESSENTIAL_DIR = os.path.join(INPUT_DIR, "essential_data")
    SUPPLEMENTAL_DIR = os.path.join(INPUT_DIR, "supplemental_data")

    # Metadata Files (Generated previously)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Image Directories
    # Using Filtered Spectrograms as per Idea 30
    IMAGE_DIR = os.path.join(SUPPLEMENTAL_DIR, "filtered_spectrograms")

    # Working & Cache Directories
    # Specific cache directory for this idea
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_30")
    SUBMISSION_DIR = "./submission"

    # Ensure directories exist (except input which is read-only)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Data Configuration
    # =========================================================================
    NUM_CLASSES = 19

    # Input Dimensions: 224 (Freq) x 448 (Time)
    # Aspect ratio 1:2 preserves temporal fidelity
    IMG_HEIGHT = 224
    IMG_WIDTH = 448
    CHANNELS = 3  # Pseudo-RGB

    # Data Loading
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # The full ensemble consists of these three backbones
    BACKBONES = ["resnet18", "efficientnet_b0", "densenet121"]

    # Anchors are trained in Phase 1 to generate soft targets
    ANCHOR_BACKBONES = ["resnet18", "efficientnet_b0"]

    # Head Configuration
    DROPOUT_RATES = [0.0, 0.1, 0.2, 0.3, 0.4]  # For Multi-Sample Dropout

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    N_FOLDS = 5
    EPOCHS = 25  # Max epochs (Early stopping will likely trigger sooner)

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler
    # Constant schedule as per idea description
    SCHEDULER_TYPE = "constant"

    # Loss Weights
    # Distillation: L = BCE(Target) + Lambda * BCE(Soft_Target)
    DISTILLATION_LAMBDA = 1.0

    # Augmentation
    MIXUP_ALPHA = 0.4

    # =========================================================================
    # Inference & TTA
    # =========================================================================
    # Cyclic TTA: Original + 3 Time-Shifts (0%, 25%, 50%, 75% implied by logic)
    TTA_STEPS = 4

    # =========================================================================
    # Device
    # =========================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def print_config(cls):
        print(f"Configuration for {cls.PROJECT_NAME}:")
        print(f"  Device: {cls.DEVICE}")
        print(f"  Image Size: {cls.IMG_HEIGHT}x{cls.IMG_WIDTH}")
        print(f"  Backbones: {cls.BACKBONES}")
        print(f"  Anchors: {cls.ANCHOR_BACKBONES}")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print(f"  Learning Rate: {cls.LEARNING_RATE}")
        print(f"  Cache Dir: {cls.CACHE_DIR}")
