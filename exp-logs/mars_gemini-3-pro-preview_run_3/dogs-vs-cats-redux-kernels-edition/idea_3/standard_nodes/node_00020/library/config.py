import os
import torch


class Config:
    """
    Configuration class for the Dog vs Cat classification task.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # ==========================================
    # Project & Paths
    # ==========================================
    PROJECT_NAME = "idea_3"

    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", PROJECT_NAME)
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    IMG_SIZE = 256  # Increased resolution for finer details
    BATCH_SIZE = 64  # Reduced batch size to prevent OOM
    NUM_WORKERS = 4  # Efficient data loading

    # ==========================================
    # Model Configuration
    # ==========================================
    # Heterogeneous Ensemble:
    # 1. ResNet50 with modern A1 recipe weights
    # 2. ConvNeXt Small with original Facebook weights (Increased capacity)
    # 3. EfficientNet V2 Small (Distinct inductive bias)
    MODEL_ARCHS = ["resnet50.a1_in1k", "convnext_small.fb_in1k", "tf_efficientnetv2_s"]

    # ==========================================
    # Training Configuration
    # ==========================================
    EPOCHS = 10  # Increased epochs to account for larger batch size
    LEARNING_RATE = 1e-4  # Standard fine-tuning rate
    WEIGHT_DECAY = 1e-2  # For AdamW
    SEED = 42  # Fixed seed for reproducibility

    # Device handling
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Inference Configuration
    # ==========================================
    TTA_FLIP = True  # Enable Horizontal Flip Test Time Augmentation

    # ==========================================
    # Debugging / Development
    # ==========================================
    DEBUG = False  # Set to True to run on a subset
    DEBUG_SAMPLES = 100  # Number of samples to use in debug mode

    @classmethod
    def setup(cls):
        """
        Creates the necessary working and submission directories if they do not exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically setup directories when config is imported
Config.setup()
