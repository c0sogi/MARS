import os
import torch


class Config:
    """
    Centralized configuration for Hotel Identification Task (Idea 4).
    Includes paths, model hyperparameters, training settings, and inference configs.
    """

    # --------------------------------------------------------------------------
    # Directory & File Paths
    # --------------------------------------------------------------------------
    # Root Input Directory (Read-Only)
    INPUT_DIR = "./input"

    # Metadata Directory (Generated CSVs)
    METADATA_DIR = "./metadata"

    # Specific Image Directories (for reference, though CSVs contain relative paths)
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Metadata File Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory for this experiment (Read/Write)
    # Used for saving models, cached data, and logs
    WORKING_DIR = "./working/idea_4"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Model Checkpoint Path
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Configuration
    # --------------------------------------------------------------------------
    NUM_CLASSES = 7770  # Total unique hotels in training set
    IMAGE_SIZE = 224  # Input size for the model (after resize/crop)
    RESIZE_SIZE = 256  # Initial resize before cropping
    NUM_WORKERS = 4  # Number of dataloader workers (12 vCPUs available)

    # Debugging / Development Flags
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLE_SIZE = 1000  # Number of samples to use in debug mode

    # --------------------------------------------------------------------------
    # Model Configuration
    # --------------------------------------------------------------------------
    BACKBONE_NAME = "efficientnet_b0"
    EMBEDDING_DIM = 512  # Dimension of the feature embedding
    PRETRAINED = True  # Use ImageNet pretrained weights

    # ArcFace Head Hyperparameters
    ARCFACE_MARGIN = 0.50
    ARCFACE_SCALE = 30.0
    ARCFACE_EASY_MARGIN = False
    ARCFACE_LS_EPS = 0.0  # Label smoothing epsilon (optional)

    # --------------------------------------------------------------------------
    # Training Configuration
    # --------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH_SIZE = 64  # Suitable for A100 with EfficientNet-B0
    NUM_EPOCHS = 12  # Sufficient for ArcFace convergence

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = NUM_EPOCHS  # Cycle length for CosineAnnealingLR
    MIN_LR = 1e-6  # Minimum learning rate

    # --------------------------------------------------------------------------
    # Inference Configuration
    # --------------------------------------------------------------------------
    TOP_K = 5  # Number of predictions per image (MAP@5)
    USE_TTA = True  # Use Test-Time Augmentation (Horizontal Flip)
