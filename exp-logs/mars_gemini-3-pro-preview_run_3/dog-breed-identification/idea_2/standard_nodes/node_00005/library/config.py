import os
import torch


class Config:
    """
    Global configuration for the Dog Breed Classification task.
    Implements settings for ConvNeXt-Tiny model, data processing, and training pipeline.
    """

    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to use a small subset of data for debugging
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use when DEBUG is True

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Cache directory for storing processed features or intermediate files (Idea 2)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_2")

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Submission Path
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "convnext_tiny"
    NUM_CLASSES = 120
    PRETRAINED = True

    # ==========================================
    # Data Preprocessing
    # ==========================================
    # Target input size for the model
    IMG_SIZE = 224

    # Resize dimension for Validation/Test before CenterCrop
    RESIZE_SIZE = 256

    # Normalization constants (ImageNet defaults)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # ==========================================
    # Compute Resources
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Utilizing available vCPUs

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Batch size optimized for A100-40GB
    BATCH_SIZE = 64

    # Phase 1: Warm-up (Frozen Backbone, Train Head only)
    WARMUP_EPOCHS = 1
    WARMUP_LR = 1e-3

    # Phase 2: Fine-tuning (Unfreeze Backbone)
    FINE_TUNE_EPOCHS = 50
    FINE_TUNE_LR = 1e-5
    WEIGHT_DECAY = 1e-4

    # Optimization Strategy
    PATIENCE = 7  # Early stopping patience

    # ==========================================
    # Setup Utility
    # ==========================================
    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories upon module import
Config.setup()
