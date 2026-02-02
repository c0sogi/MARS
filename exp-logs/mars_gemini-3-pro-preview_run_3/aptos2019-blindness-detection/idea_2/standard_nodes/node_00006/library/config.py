import os
import torch


class Config:
    """
    Configuration class for the High-Resolution EfficientNet Regression task.
    """

    # ==========================================
    # General Configuration
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to train on a small subset for debugging
    DEBUG_SUBSET_SIZE = 100  # Number of samples to use when DEBUG is True

    # ==========================================
    # Directories and Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Output directory for the current idea
    OUTPUT_DIR = os.path.join(WORKING_DIR, "idea_3")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Metadata CSV paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Submission path
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    IMG_SIZE = 300  # Increased resolution for EfficientNet-B3
    NUM_CLASSES = 1  # Regression output (scalar)

    # Normalization constants (ImageNet defaults)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # ==========================================
    # Model Configuration
    # ==========================================
    MODEL_NAME = "efficientnet_b3"
    PRETRAINED = True
    DROPOUT_RATE = 0.3  # Regularization before the final linear layer

    # ==========================================
    # Training Configuration
    # ==========================================
    BATCH_SIZE = 32
    EPOCHS = 12  # Sufficient for convergence with pre-trained weights
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Compute settings
    NUM_WORKERS = 4  # Optimized for the 12 vCPU environment
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP = True  # Enable Automatic Mixed Precision (FP16)

    # Loss function (MSE for regression)
    LOSS_FN = "MSE"

    # Early Stopping
    PATIENCE = 3

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 30)
        print("CONFIG")
        print("=" * 30)
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k}: {v}")
        print("=" * 30)
