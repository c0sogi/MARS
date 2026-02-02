import os
import random
import numpy as np
import torch


class Config:
    """
    Configuration class for the Hybrid Neuro-Symbolic Text Normalization model.
    Stores file paths, hyperparameters, and special token definitions.
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    @staticmethod
    def set_seed(seed=SEED):
        """Sets the random seed for reproducibility across all libraries."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

    # ==========================================
    # File Paths & Directories
    # ==========================================
    # Input Metadata
    METADATA_DIR = "./metadata"
    TRAIN_DATA = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA = os.path.join(METADATA_DIR, "test.parquet")

    # Working Directory (Cache & Model Artifacts)
    WORKING_DIR = "./working/idea_2"
    CACHE_DIR = WORKING_DIR

    # Model Checkpoint
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "seq2seq_best_model.pt")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Processing & Vocabulary
    # ==========================================
    # Special Tokens
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"
    SEP_TOKEN = "<SEP>"  # Separator for context: prev <SEP> curr <SEP> next

    # Reserved Indices
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3
    SEP_IDX = 4

    # Sequence Constraints
    MAX_SEQ_LEN = 128  # Maximum length for input/output character sequences
    CONTEXT_WINDOW = 1  # Number of neighbor tokens to include (1 left, 1 right)

    # Symbolic Lookup Config
    NGRAM_ORDER = 3  # Trigram priority

    # ==========================================
    # Model Hyperparameters (Seq2Seq)
    # ==========================================
    EMBED_DIM = 128
    HIDDEN_DIM = 256
    NUM_LAYERS = 2  # Number of GRU/LSTM layers
    DROPOUT = 0.2

    # ==========================================
    # Training Settings
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 15
    PATIENCE = 3  # Early stopping patience
    CLIP_GRAD = 1.0  # Gradient clipping value
    TEACHER_FORCING_RATIO = 0.5
