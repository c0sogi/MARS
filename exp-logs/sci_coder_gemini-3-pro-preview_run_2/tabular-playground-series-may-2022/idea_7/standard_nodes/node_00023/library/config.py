import os
import torch
import numpy as np
import random


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ModelConfig:
    """
    Configuration class for the II-ResFunnel-GLU pipeline.
    """

    # --------------------------------------------------------------------------
    # General Setup
    # --------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Adjust based on vCPU availability (12 vCPUs available)

    # --------------------------------------------------------------------------
    # File Paths & Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for caching processed data and saving models
    WORKING_DIR = "./working/idea_7"

    # Raw Data Files
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files
    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Files
    PROCESSED_DATA_PATH = os.path.join(WORKING_DIR, "processed_data.npz")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = "./submission/submission.csv"

    # --------------------------------------------------------------------------
    # Data Parameters
    # --------------------------------------------------------------------------
    # f_27 string feature length
    SEQUENCE_LENGTH = 10
    # Number of continuous features (f_00 to f_30, excluding f_27)
    NUM_CONT_FEATURES = 30
    # Vocabulary size for character embeddings (A-Z + potential padding/unknown)
    # 26 uppercase letters. We'll set a safe upper bound or calculate dynamically.
    # Usually indices are 1-26, 0 for padding.
    VOCAB_SIZE = 30

    # --------------------------------------------------------------------------
    # Model Architecture (II-ResFunnel-GLU)
    # --------------------------------------------------------------------------
    # Embedding dimension for each character in f_27
    EMBED_DIM = 32

    # Funnel Stages dimensions: Stage 1 -> Stage 2 -> Stage 3
    # As per description: 512 -> 256 -> 128
    HIDDEN_DIMS = [512, 256, 128]

    # Regularization
    DROPOUT_RATE = 0.35

    # Input Injection
    # We inject raw input at the start of Stage 2 and Stage 3
    INJECT_AT_STAGES = [
        1,
        2,
    ]  # Indices of stages where injection happens (0-based index)

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 2048  # Large batch size for tabular data on A100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.02
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10

    # --------------------------------------------------------------------------
    # Debugging / Development
    # --------------------------------------------------------------------------
    # Set to True to use a smaller subset of data for rapid prototyping
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 10000

    @classmethod
    def create_dirs(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.SUBMISSION_PATH), exist_ok=True)


# Ensure directories exist upon import or usage
ModelConfig.create_dirs()
