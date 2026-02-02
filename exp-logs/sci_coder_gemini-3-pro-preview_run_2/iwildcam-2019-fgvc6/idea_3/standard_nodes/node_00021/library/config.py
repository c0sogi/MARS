import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    PROJECT_NAME = "Animal_Classification_EfficientNetB3"
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Directory Paths
    # =========================================================================
    # Input Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working Directory for this specific idea (Idea 3)
    WORKING_DIR = "./working/idea_3"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    IMAGE_SIZE = 300  # Native resolution for EfficientNet-B3
    NUM_CLASSES = 23
    BATCH_SIZE = 64
    NUM_WORKERS = 12  # Matches available vCPUs

    # Normalization constants (ImageNet defaults)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # =========================================================================
    # Model Configuration
    # =========================================================================
    MODEL_NAME = "efficientnet_b3"
    PRETRAINED = True
    DROPOUT_RATE = 0.3

    # =========================================================================
    # Training Configuration (Two-Stage Strategy)
    # =========================================================================
    # Stage 1: Aggressive Head Alignment (Frozen Backbone)
    LR_STAGE1 = 1e-3
    EPOCHS_STAGE1 = 6

    # Stage 2: Fine-Tuning (Unfrozen Top Blocks)
    LR_STAGE2 = 1e-4
    EPOCHS_STAGE2 = 10
    WEIGHT_DECAY = 1e-4

    # Scheduler
    SCHEDULER_T_MAX = EPOCHS_STAGE2  # For Cosine Annealing

    # Early Stopping
    PATIENCE = 5

    # Debugging/Development
    DEBUG = False  # Set to True to run on a small subset of data
    DEBUG_SAMPLE_SIZE = 1000
