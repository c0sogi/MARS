import os
import random
import torch
import numpy as np


def seed_everything(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    # Experiment Metadata
    NAME = "idea_12"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use in debug mode

    # Paths
    ROOT_DIR = "."
    INPUT_DIR = os.path.join(ROOT_DIR, "input")
    METADATA_DIR = os.path.join(ROOT_DIR, "metadata")
    WORKING_DIR = os.path.join(ROOT_DIR, "working", NAME)
    SUBMISSION_DIR = os.path.join(ROOT_DIR, "submission")

    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Architecture
    MODEL_NAME = "microsoft/deberta-v3-large"
    MAX_LEN = 128
    HIDDEN_SIZE = 1024  # DeBERTa-v3-large hidden size

    # Head Specifics (Weighted Layer Pooling + CNN)
    N_POOLING_LAYERS = 4
    CNN_KERNEL_SIZE = 3
    CNN_OUT_CHANNELS = 256

    # Training Hyperparameters
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 16
    EPOCHS = 4
    LEARNING_RATE = 5e-5  # Higher base LR to compensate for LLRD
    LLRD_DECAY = 0.9  # Layer-wise Learning Rate Decay
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    WARMUP_RATIO = 0.1
    USE_AMP = True  # Automatic Mixed Precision

    # Loss & Targets
    SIGMA = 1.0  # Gaussian smoothing sigma for soft targets
    TRAIN_EXCLUDE_NEUTRAL = True  # Exclude neutral tweets from training

    # Cross-Validation
    N_FOLDS = 5

    # System / Hardware
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TOKENIZERS_PARALLELISM = "false"

    @classmethod
    def setup(cls):
        """Creates necessary directories and sets seeds."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set environment variables
        os.environ["TOKENIZERS_PARALLELISM"] = cls.TOKENIZERS_PARALLELISM

        # Set seeds
        seed_everything(cls.SEED)

    @classmethod
    def get_tokenizer_path(cls):
        return cls.MODEL_NAME
