import os
import torch


class Config:
    """
    Configuration class for High-Resolution Heterogeneous Ensemble with Model EMA.
    """

    # -------------------------------------------------------------------------
    # General Settings
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Directories & Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Working directory for caching and model checkpoints
    WORKING_DIR = "./working/idea_11"
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata file paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    CLASSES = ["healthy", "multiple_diseases", "rust", "scab"]
    NUM_CLASSES = len(CLASSES)

    # -------------------------------------------------------------------------
    # Model Architectures
    # -------------------------------------------------------------------------
    # Backbone 1: EfficientNetV2-M (High-Res Texture Expert)
    # Uses Fused-MBConv blocks, good for high-freq details.
    MODEL_1_NAME = "tf_efficientnetv2_m"
    MODEL_1_IMG_SIZE = 512

    # Backbone 2: MaxViT-Small (High-Res Global Expert)
    # Uses Multi-Axis Attention (Block + Grid).
    MODEL_2_NAME = "maxvit_small_tf_384"
    MODEL_2_IMG_SIZE = 384

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 16  # Fits A100 40GB with AMP and high resolutions
    EPOCHS = 25
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Optimizer & Scheduler
    # AdamW + Cosine Annealing
    T_MAX = EPOCHS
    MIN_LR = 1e-6

    # Early Stopping
    # Relaxed patience to allow EMA to stabilize
    PATIENCE = 10

    # Model Exponential Moving Average (EMA)
    USE_EMA = True
    EMA_DECAY = 0.999

    # Loss Function
    # CrossEntropy with Inverse Frequency Class Weights
    USE_CLASS_WEIGHTS = True

    # Precision
    USE_AMP = True

    # -------------------------------------------------------------------------
    # Augmentation Strategy
    # -------------------------------------------------------------------------
    # Strong Geometric Augmentations (Shift, Scale, Rotate, Flip)
    # Excludes Cutout and Photometric distortions
    AUG_SHIFT_LIMIT = 0.1
    AUG_SCALE_LIMIT = 0.2
    AUG_ROTATE_LIMIT = 15
    AUG_PROB = 0.5

    # -------------------------------------------------------------------------
    # Inference Strategy
    # -------------------------------------------------------------------------
    # Test-Time Augmentation (TTA)
    # Only Horizontal Flip is used; Vertical Flip/Transpose excluded
    TTA_HORIZONTAL_FLIP = True
