import os
import torch


class Config:
    """
    Configuration class for Cassava Leaf Disease Classification (Idea 8).
    Implements a Precision-Optimized Heterogeneous Ensemble strategy.
    """

    # ====================================================
    # General Setup
    # ====================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 100

    # Compute environment
    NUM_WORKERS = 8  # Optimized for 12 vCPUs
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====================================================
    # Directories & Paths
    # ====================================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Image Directories
    TRAIN_IMAGES_DIR = os.path.join(INPUT_ROOT, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_ROOT, "test_images")

    # Metadata Paths (Pre-generated)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory for Idea 8 (Cache & Checkpoints)
    WORKING_DIR = "./working/idea_8"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission Output
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ====================================================
    # Data Configuration
    # ====================================================
    IMG_SIZE = 384  # Unified resolution for both models
    NUM_CLASSES = 5

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    EPOCHS = 15
    BATCH_SIZE = 32  # Optimized for A100-40GB with AMP

    # Optimization
    AMP = True  # Automatic Mixed Precision
    MAX_GRAD_NORM = 1.0
    LABEL_SMOOTHING = 0.1

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    MIN_LR = 1e-6

    # Early Stopping
    PATIENCE = 3

    # ====================================================
    # Model Architecture & Specifics
    # ====================================================

    # Model A: Global Expert (Vision Transformer)
    # Captures long-range dependencies
    MODEL_A_NAME = "vit_base_patch16_384"
    MODEL_A_LR = 2e-5  # Conservative learning rate
    MODEL_A_WEIGHT_DECAY = 0.01

    # Model B: Local Expert (CNN)
    # Captures high-frequency textures and lesion boundaries
    MODEL_B_NAME = "tf_efficientnet_b4_ns"
    MODEL_B_LR = 1e-4
    MODEL_B_WEIGHT_DECAY = 1e-5

    # ====================================================
    # Augmentation Strategy
    # ====================================================
    # Geometric Augmentations (Full Dihedral Group D4 + Affine)
    # Includes: HorizontalFlip, VerticalFlip, Transpose, ShiftScaleRotate
    USE_GEOMETRIC_AUG = True

    # Mixing Regularization
    USE_MIXING = True
    MIXUP_ALPHA = 0.2
    CUTMIX_ALPHA = 1.0
    MIX_PROB = 0.5

    # ====================================================
    # Inference & Ensembling
    # ====================================================
    USE_TTA = True  # Test-Time Augmentation
    TTA_STEPS = 4  # Number of TTA views (e.g., Original + Flips/Transpose)
