import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class Config:
    """
    Configuration class for the Text Normalization project.
    Contains paths, model hyperparameters, training settings, and data processing constants.
    """

    # ==========================================
    # General Configuration
    # ==========================================
    SEED = 42
    PROJECT_NAME = "TextNormalization_Idea2"

    # ==========================================
    # File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"

    # Data Sources (from Metadata)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Artifacts & Caching
    MODEL_CHECKPOINT = os.path.join(WORKING_DIR, "seq2seq_best.pth")
    VOCAB_PATH = os.path.join(WORKING_DIR, "vocab.json")
    HFBB_CACHE_DIR = os.path.join(WORKING_DIR, "hfbb_cache")

    # ==========================================
    # Data Processing & Tokenization
    # ==========================================
    # Special Tokens for Character-Level Tokenizer
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"
    SEP_TOKEN = "<sep>"

    # Token Indices
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3
    SEP_IDX = 4

    SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, SEP_TOKEN]

    # Sequence Constraints
    MAX_SEQ_LEN = 128  # Maximum length for character sequences (input/output)
    CONTEXT_WINDOW = 1  # Number of words to include as context (left and right)

    # ==========================================
    # Model Architecture (Seq2Seq Transformer)
    # ==========================================
    EMBED_DIM = 256
    HIDDEN_DIM = 512
    N_LAYERS = 4  # Number of encoder and decoder layers
    N_HEADS = 4  # Number of attention heads
    DROPOUT = 0.1
    FORWARD_EXPANSION = 2  # Multiplier for FeedForward network dimension

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 256  # Large batch size for A100
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 15
    EARLY_STOPPING_PATIENCE = 3
    WARMUP_STEPS = 1000
    CLIP_GRAD = 1.0  # Gradient clipping value

    # ==========================================
    # Inference & Evaluation
    # ==========================================
    BEAM_WIDTH = 3  # Beam search width for generation

    # ==========================================
    # Compute & Runtime
    # ==========================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Debugging
    # ==========================================
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SIZE = 10000  # Number of samples to use when DEBUG is True

    @classmethod
    def setup_directories(cls):
        """
        Creates necessary directories for working files and submissions.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.HFBB_CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(
            f"Directories initialized: {cls.WORKING_DIR}, {cls.HFBB_CACHE_DIR}, {cls.SUBMISSION_DIR}"
        )
