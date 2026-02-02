import os
import torch


class Config:
    """
    Configuration class for Dog vs Cat Classification using EfficientNet-B0
    and K-Fold Cross Validation.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True for fast debugging on a subset
    NUM_WORKERS = 4  # Number of data loading workers

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_CSV = os.path.join(METADATA_DIR, "test.csv")

    WORKING_DIR = "./working"
    # Specific directory for this idea/experiment to serve as cache and output
    IDEA_DIR = os.path.join(WORKING_DIR, "idea_2")
    CHECKPOINT_DIR = os.path.join(IDEA_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create necessary directories
    os.makedirs(IDEA_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Model Architecture
    # =========================================================================
    MODEL_NAME = "efficientnet_b0"
    PRETRAINED = True
    NUM_CLASSES = 1  # Binary classification (Dog vs Cat)

    # =========================================================================
    # Data Preprocessing & Augmentation
    # =========================================================================
    IMG_SIZE = 224
    IMG_MEAN = [0.485, 0.456, 0.406]  # ImageNet statistics
    IMG_STD = [0.229, 0.224, 0.225]

    # Augmentation Hyperparameters
    AUG_CROP_SCALE = (0.8, 1.0)
    AUG_HFLIP_PROB = 0.5
    AUG_COLOR_JITTER = 0.1  # Brightness, contrast, saturation, hue

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    N_FOLDS = 5
    EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2  # For AdamW
    EARLY_STOPPING_PATIENCE = 3

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # =========================================================================
    # Compute
    # =========================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
