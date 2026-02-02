import os
import torch


class Config:
    """
    Configuration for the Stabilized High-Resolution Multi-Task Network (SHR-MTN).
    """

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_9"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Submission
    # Ensuring submission is saved to the required location
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    # High resolution for detecting microcalcifications
    IMAGE_SIZE = (1024, 1024)
    NUM_CHANNELS = 3

    # Debugging / Development
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 500  # Number of samples to use when DEBUG is True

    # ==========================================
    # Model Architecture
    # ==========================================
    # EfficientNetV2-Small: Good balance of speed/accuracy for high-res
    MODEL_NAME = "tf_efficientnetv2_s"
    PRETRAINED = True

    # Multi-Task Learning Heads
    USE_AUX_HEADS = True
    NUM_CANCER_CLASSES = 1  # Binary Classification
    NUM_BIRADS_CLASSES = 1  # Regression (MSE)
    NUM_DENSITY_CLASSES = 4  # Classification (CrossEntropy)

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    NUM_EPOCHS = 5

    # Batch size of 8 fits comfortably on A100-40GB with 1024x1024 resolution
    BATCH_SIZE = 8

    # Optimizer
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Loss Function Configuration
    # Pos weight of 15.0 to counter the ~2% cancer prevalence
    POS_WEIGHT = 15.0

    # Initialization
    # Bias init to log(0.02/0.98) ~ -3.9 to prevent initial loss explosion
    INIT_BIAS = -3.9

    # ==========================================
    # Hardware & Compute
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Utilizing all 12 vCPUs
    PIN_MEMORY = True

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
Config.setup()
