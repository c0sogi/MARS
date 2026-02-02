import os
import torch


class Config:
    # =========================================================================
    # General Settings & Reproducibility
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 100

    # =========================================================================
    # Directory Configuration
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_16"

    # Ensure the working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Data Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Output Paths
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # Data Parameters
    # =========================================================================
    LABELS = ["healthy", "multiple_diseases", "rust", "scab"]
    NUM_CLASSES = len(LABELS)
    IMG_SIZE = (384, 384)  # Resolution for fine-grained disease detection

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Heterogeneous Ensemble: Texture Expert (CNN) + Context Expert (ConvNeXt)
    # Cite debug_lesson_8: Using stable base names. Swin removed due to incompatibility with features_only.
    BACKBONES = ["tf_efficientnetv2_m", "convnext_tiny"]

    # Head Configuration
    POOLING_TYPE = "gem"  # Generalized Mean Pooling
    GEM_P = 3.0  # Initial power parameter for GeM

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    N_FOLDS = 5
    EPOCHS = 25  # Max epochs, controlled by Early Stopping
    BATCH_SIZE = 16  # Conservative batch size for A100 + 384px + Large Models

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    MIN_LR = 1e-6

    # Regularization
    PATIENCE = 10  # Relaxed patience to allow Swin Transformer to converge

    # =========================================================================
    # Advanced Training & Inference Techniques
    # =========================================================================
    USE_AMP = True  # Automatic Mixed Precision

    # Model EMA (Exponential Moving Average)
    USE_EMA = True
    EMA_DECAY = 0.999
    EMA_UPDATE_EVERY = 1

    # Test Time Augmentation (TTA)
    # Strategy: Rotational Invariance (Original, HFlip, VFlip, Transpose)
    USE_TTA = True

    # =========================================================================
    # Hardware
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for the 12 vCPU environment
