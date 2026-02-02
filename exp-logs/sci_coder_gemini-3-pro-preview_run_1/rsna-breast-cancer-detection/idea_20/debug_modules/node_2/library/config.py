import os
import torch


class Config:
    """
    Central configuration for the Breast Cancer Detection task.
    Implements the 'Channel-Attentive Symmetry-Difference Siamese Network' parameters.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data
    DEBUG_SAMPLE_SIZE = 500  # Number of samples if DEBUG is True

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Cache directory for this specific idea (Idea 20)
    # Using parquet/npy instead of pickle as requested
    IDEA_CACHE_DIR = os.path.join(WORKING_DIR, "idea_20")

    # Metadata files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Raw Image Directories
    # Note: Images are in [train/test]_images/[patient_id]/[image_id].dcm
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Submission
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # Data Processing Parameters
    # =========================================================================
    # Input dimensions: (3, 768, 768)
    # Channels: [Mammogram Pixel Data, Spatially Broadcasted Age, Spatially Broadcasted Implant]
    IMG_SIZE = (768, 768)
    IN_CHANNELS = 3

    # Normalization statistics can be defined here if calculated globally,
    # otherwise handled in the dataset class.

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    MODEL_NAME = "tf_efficientnet_b2_ns"

    # Loss Function Weights
    # Calculated as approx inverse class frequency (Negative / Positive) ~ 47.0
    POS_WEIGHT = 47.0

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Batch size 8 chosen for 768x768 resolution + Siamese (2x images) on A100
    BATCH_SIZE = 8
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Gradient accumulation steps can be increased if BATCH_SIZE must be lowered
    ACCUMULATION_STEPS = 1

    # Early Stopping
    PATIENCE = 3

    # =========================================================================
    # Hardware
    # =========================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Optimized for 12 vCPUs
    PIN_MEMORY = False

    @classmethod
    def setup(cls):
        """
        Ensures necessary working directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.IDEA_CACHE_DIR, exist_ok=True)
