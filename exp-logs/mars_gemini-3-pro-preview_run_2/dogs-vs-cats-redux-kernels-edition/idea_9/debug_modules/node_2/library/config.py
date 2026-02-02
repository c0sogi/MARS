import os
import torch


class Config:
    """
    Configuration class for the Dog vs Cat Heterogeneous Ensemble Task.
    Defines hyperparameters, file paths, and model settings.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SUBSET_SIZE = 100  # Number of samples if DEBUG is True

    # =========================================================================
    # Compute
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = os.cpu_count()  # Use available vCPUs

    # =========================================================================
    # Data Paths
    # =========================================================================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # =========================================================================
    # Output Paths
    # =========================================================================
    # Specific working directory for this idea/experiment
    WORKING_DIR = "./working/idea_9"

    # Directory to save model checkpoints
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Directory to save final submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Model Architecture (Heterogeneous Ensemble)
    # =========================================================================
    # We use a mix of CNN (ConvNeXt) and Transformer (Swin) architectures
    # to leverage different inductive biases.
    MODEL_ARCHS = [
        "convnext_small.fb_in22k",  # CNN: Good local features
        "swin_small_patch4_window7_224",  # Transformer: Good global context
    ]

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    IMG_SIZE = 224  # Resolution fixed to 224x224
    BATCH_SIZE = 32  # Safe batch size for A100 with these models
    EPOCHS = 20  # Long training duration for convergence

    # Optimizer settings
    LEARNING_RATE = 1e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-2

    # =========================================================================
    # Cross Validation
    # =========================================================================
    N_FOLDS = 5  # 5-Fold Stratified Cross-Validation

    # =========================================================================
    # Regularization & Augmentation
    # =========================================================================
    MIXUP_ALPHA = 0.2  # Mixup strength
    CROP_SCALE_MIN = (
        0.8  # Minimum scale for RandomResizedCrop (avoid aggressive cropping)
    )

    # =========================================================================
    # Inference / TTA
    # =========================================================================
    TTA_FLIP = True  # Use Horizontal Flip Test-Time Augmentation
