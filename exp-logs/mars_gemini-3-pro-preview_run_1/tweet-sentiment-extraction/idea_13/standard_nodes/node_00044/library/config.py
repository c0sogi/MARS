import os
import torch


class Config:
    # General Settings
    SEED = 42
    DEBUG = False  # Set to True for fast debugging on a subset
    N_FOLDS = 5
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    # Using metadata files as per instructions
    TRAIN_FILE = "./metadata/train_metadata.csv"
    TEST_FILE = "./metadata/test_metadata.csv"
    SAMPLE_SUBMISSION = "./input/sample_submission.csv"

    # Working Directory for Caching and Models
    # Specifically using idea_14 as the iteration identifier
    WORKING_DIR = "./working/idea_14/"
    OUTPUT_DIR = "./working/idea_14/"
    SUBMISSION_DIR = "./submission/"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Architecture
    MODEL_PATH = "microsoft/deberta-v3-large"
    # Max length 128 covers the max char length (141) + special tokens comfortably
    MAX_LEN = 128

    # Training Hyperparameters
    EPOCHS = 5
    TRAIN_BATCH_SIZE = 8  # Adjusted for A100 40GB + Large model
    VALID_BATCH_SIZE = 16
    ACCUMULATION_STEPS = 1

    # Optimization
    LEARNING_RATE = 2e-5  # Base learning rate
    LLRD_DECAY = 0.9  # Layer-wise Learning Rate Decay factor
    WEIGHT_DECAY = 0.01
    EPS = 1e-6
    SCHEDULER_TYPE = "cosine"
    WARMUP_RATIO = 0.1
    CLIP_GRAD_NORM = 1.0

    # Model Specifics
    HIDDEN_SIZE = 1024  # DeBERTa-v3-large hidden size
    DROP_RATE = 0.1  # Standard dropout for the head
    USE_CNN_HEAD = True  # Toggle for Shared CNN Head
    CNN_KERNEL_SIZE = 3

    # Loss Function Hyperparameters
    # Total Loss = (1 - ALPHA) * KL_Loss + ALPHA * Soft_Jaccard_Loss
    LOSS_ALPHA = 0.5
    LABEL_SMOOTHING_SIGMA = 1.0  # Sigma for Gaussian smoothing of targets

    # Caching
    LOAD_CACHED_DATA = True  # Flag to control data loading logic in processing scripts

    @classmethod
    def setup(cls):
        """Ensures necessary directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories immediately upon import
Config.setup()
