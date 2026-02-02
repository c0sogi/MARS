import os
import random
import numpy as np
import torch


class Config:
    """
    Configuration class for Text Normalization Task.
    Stores all hyperparameters, file paths, and settings.
    """

    # ==========================================
    # File Paths and Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")
    SUBMISSION_DIR = "./submission"

    # Input Metadata Files
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.csv")
    VAL_FILE = os.path.join(METADATA_DIR, "val.csv")
    TEST_FILE = os.path.join(METADATA_DIR, "test.csv")

    # Cache Files (Processed Data)
    TRAIN_CACHE = os.path.join(CACHE_DIR, "train_processed.parquet")
    VAL_CACHE = os.path.join(CACHE_DIR, "val_processed.parquet")
    TEST_CACHE = os.path.join(CACHE_DIR, "test_processed.parquet")
    VOCAB_CACHE = os.path.join(CACHE_DIR, "vocab.npy")

    # Output Files
    MODEL_SAVE_PATH = os.path.join(CACHE_DIR, "best_model.pt")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Special Tokens
    # ==========================================
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"
    SEP_TOKEN = "|"  # Separator for context: prev|curr|next

    # Reserved Indices
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    MAX_SEQ_LEN = 128  # Maximum length for character sequences (input/output)
    CONTEXT_WINDOW = 1  # Number of tokens to include as context (left and right)

    # Balancing Strategy:
    # Downsample 'PLAIN' class tokens where before == after (no change).
    # This forces the model to focus on normalization tasks (Dates, Numbers, etc.)
    # A ratio of 0.05 keeps ~5% of the unchanged PLAIN tokens.
    PLAIN_DOWNSAMPLE_RATIO = 0.05

    # Debugging
    DEBUG = False  # Set to True to run on a small subset for quick testing
    DEBUG_SIZE = 50000  # Number of samples if DEBUG is True

    # ==========================================
    # Model Hyperparameters (Seq2Seq LSTM + Attention)
    # ==========================================
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HIDDEN_DIM = 512
    ENC_DROPOUT = 0.2
    DEC_DROPOUT = 0.2
    N_LAYERS = 1  # Number of LSTM layers

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 1024  # Large batch size for A100 GPU
    LEARNING_RATE = 0.001
    N_EPOCHS = 15
    TEACHER_FORCING_RATIO = 0.5
    CLIP_GRAD = 1.0  # Gradient clipping threshold
    PATIENCE = 3  # Early stopping patience (epochs without improvement)

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def set_seed(cls):
        """Sets random seeds for full reproducibility."""
        seed = cls.SEED
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

    @staticmethod
    def get_device():
        """Returns the computing device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
