import os
import torch
import random
import numpy as np


class Config:
    """
    Central configuration for the Stack Exchange Tag Prediction Task.
    Defines hyperparameters, file paths, and setup utilities.
    """

    # ==========================================
    # 1. File System Paths
    # ==========================================
    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"
    SUBMISSION_DIR = "./submission"

    # Input Data Files
    # Metadata files contain Id, Tags (for train/val), and file_path
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Raw content files (referenced by metadata)
    TRAIN_RAW_PATH = os.path.join(INPUT_DIR, "train.csv")
    TEST_RAW_PATH = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache/Artifact Paths (for persistence between runs)
    TOKENIZER_PATH = os.path.join(WORKING_DIR, "tokenizer.json")
    MLB_PATH = os.path.join(WORKING_DIR, "mlb.joblib")  # MultiLabelBinarizer
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "dual_stream_cnn.pth")

    # ==========================================
    # 2. Data Processing Hyperparameters
    # ==========================================
    # Text Processing
    MAX_TITLE_LEN = 30  # Short length for concise titles
    MAX_BODY_LEN = 400  # Longer length for detailed body content
    VOCAB_SIZE = 100000  # Limit vocabulary to top frequent words
    MIN_FREQ = 2  # Minimum word frequency to be included

    # Debugging / Development
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLE_SIZE = 50000

    # ==========================================
    # 3. Model Architecture (Dual-Stream TextCNN)
    # ==========================================
    EMBED_DIM = 300

    # CNN Architecture
    # Distinct kernel sizes for Title (short patterns) and Body (longer contexts)
    TITLE_KERNELS = [2, 3, 4]
    BODY_KERNELS = [3, 4, 5]
    NUM_FILTERS = 128  # Filters per kernel size
    DROPOUT = 0.5  # Dropout rate for the final fully connected layer

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 2048  # Optimized for A100 GPU (40GB VRAM)
    LEARNING_RATE = 1e-3  # Initial learning rate (used with OneCycleLR)
    NUM_EPOCHS = 15  # Max epochs
    PATIENCE = 3  # Early stopping patience
    WEIGHT_DECAY = 1e-5  # Regularization
    GRAD_CLIP = 1.0  # Gradient clipping threshold

    # Reproducibility
    SEED = 42

    @classmethod
    def setup(cls):
        """
        Initializes the environment:
        1. Creates necessary working and submission directories.
        2. Sets random seeds for reproducibility across random, numpy, and torch.
        """
        # Ensure directories exist
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set random seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    @staticmethod
    def get_device():
        """Returns the appropriate PyTorch device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
