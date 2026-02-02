import os
import torch


class Config:
    """
    Centralized configuration for the Cactus Identification task.
    Includes file paths, system settings, data stats, training hyperparameters,
    and model architecture definitions.
    """

    # =========================================================================
    # System Settings
    # =========================================================================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Number of workers for data loading
    NUM_WORKERS = 4

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input directories
    INPUT_DIR = "./input"
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata paths (pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output directories
    # Working directory for checkpoints and intermediate files
    WORKING_DIR = "./working/idea_4"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Path to save the best model checkpoint
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Constants
    # =========================================================================
    IMG_SIZE = 32

    # Normalization statistics (RGB) derived from dataset analysis
    # Mean: R=128.36, G=115.25, B=119.40 -> Scaled to [0, 1]
    NORM_MEAN = [0.5034, 0.4520, 0.4683]
    # Std: R=38.60, G=35.68, B=39.15 -> Scaled to [0, 1]
    NORM_STD = [0.1514, 0.1399, 0.1536]

    # Debugging: Set to a small number (e.g., 100) to run a quick test
    DEBUG = False
    DEBUG_SAMPLES = 100

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    EPOCHS = 30
    BATCH_SIZE = 128

    # Optimizer settings (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler settings (Cosine Annealing)
    # T_max is usually set to EPOCHS
    MIN_LR = 1e-6

    # Regularization
    # Mixup alpha parameter (Beta distribution)
    MIXUP_ALPHA = 0.2

    # =========================================================================
    # Model Architecture (Custom Shallow ConvNeXt)
    # =========================================================================
    # Depths of each stage (number of blocks)
    # Keeping it shallow for 32x32 images to prevent overfitting
    MODEL_DEPTHS = [2, 2, 2, 2]

    # Channel dimensions for each stage
    MODEL_DIMS = [64, 128, 256, 512]

    # Drop path rate (stochastic depth)
    DROP_PATH_RATE = 0.0

    # =========================================================================
    # Inference
    # =========================================================================
    # Test Time Augmentation (TTA)
    USE_TTA = True
