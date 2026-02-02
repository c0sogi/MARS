import os
import torch


class Config:
    """
    Central configuration for the Dog vs Cat classification task.
    Implements the settings for the Triple Heterogeneous Ensemble strategy (Idea 6).
    """

    # -------------------------------------------------------------------------
    # Directories & Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for checkpoints, cached data, and logs
    WORKING_DIR = "./working/idea_6"

    # Directory for final submission
    SUBMISSION_DIR = "./submission"

    # Metadata CSV paths (pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Final submission file path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    IMG_SIZE = 256
    NUM_CLASSES = 1  # Binary classification (0=Cat, 1=Dog)

    # Augmentation Hyperparameters
    # Context-Preserving Augmentation: Restricted scale to avoid cropping out the subject
    CROP_SCALE = (0.8, 1.0)

    # -------------------------------------------------------------------------
    # Model Configuration
    # -------------------------------------------------------------------------
    # List of models for the Triple Heterogeneous Ensemble
    # 1. ResNet-50: Standard CNN anchor
    # 2. ConvNeXt-Small: Modernized CNN with Transformer-like blocks
    # 3. EfficientNetV2-Small: Depthwise Separable / NAS based
    MODELS = ["resnet50.a1_in1k", "convnext_small.fb_in1k", "tf_efficientnetv2_s.in1k"]

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    SEED = 42
    EPOCHS = 10
    BATCH_SIZE = 64

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # -------------------------------------------------------------------------
    # Compute & Hardware
    # -------------------------------------------------------------------------
    NUM_WORKERS = 4  # Adjusted for 12 vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # Debugging / Development
    # -------------------------------------------------------------------------
    # If True, runs training/inference on a small subset of data for quick verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 200
