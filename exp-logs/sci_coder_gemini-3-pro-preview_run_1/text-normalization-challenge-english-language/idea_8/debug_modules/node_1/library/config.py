import os
import torch
import random
import numpy as np


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of subprocesses for data loading

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_8"
    SUBMISSION_DIR = "./submission"

    # Automatically ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Metadata Files (Pre-split)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "en_sample_submission_2.csv")

    # Cache Files (Parquet/Numpy for fast loading)
    # Grouped data (sentences) for Tagger
    TRAIN_GROUPED_PATH = os.path.join(WORKING_DIR, "train_grouped.parquet")
    VAL_GROUPED_PATH = os.path.join(WORKING_DIR, "val_grouped.parquet")
    TEST_GROUPED_PATH = os.path.join(WORKING_DIR, "test_grouped.parquet")

    # Filtered data (changed tokens only) for Seq2Seq
    TRAIN_SEQ2SEQ_PATH = os.path.join(WORKING_DIR, "train_seq2seq.parquet")
    VAL_SEQ2SEQ_PATH = os.path.join(WORKING_DIR, "val_seq2seq.parquet")

    # Vocabularies and Knowledge Base
    VOCAB_TOKENS_PATH = os.path.join(WORKING_DIR, "vocab_tokens.parquet")
    VOCAB_CHARS_PATH = os.path.join(WORKING_DIR, "vocab_chars.parquet")
    VOCAB_CLASSES_PATH = os.path.join(WORKING_DIR, "vocab_classes.parquet")
    KNOWLEDGE_BASE_PATH = os.path.join(WORKING_DIR, "knowledge_base.parquet")
    CLASS_WEIGHTS_PATH = os.path.join(WORKING_DIR, "class_weights.npy")

    # Model Checkpoints
    TAGGER_MODEL_PATH = os.path.join(WORKING_DIR, "tagger_best_model.pth")
    SEQ2SEQ_MODEL_PATH = os.path.join(WORKING_DIR, "seq2seq_best_model.pth")

    # Final Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing Parameters
    # ==========================================
    # Vocabulary Construction
    MAX_TOKEN_VOCAB_SIZE = 80000  # Large vocabulary to minimize UNKs
    MIN_TOKEN_FREQ = 2  # Minimum frequency to include a token

    # Sequence Lengths
    MAX_CHAR_LEN = 30  # Max characters per token (for Tagger CNN)
    MAX_SEQ_LEN = 128  # Max sequence length for Seq2Seq generation

    # Special Tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"

    # Debugging / Development
    DEBUG = False  # Set to True to use a small subset of data
    DEBUG_SIZE = 50000  # Number of sentences/tokens to use if DEBUG is True

    # ==========================================
    # Model Hyperparameters
    # ==========================================

    # 1. Bi-LSTM Tagger (Morphologically Aware)
    TAGGER_EMBED_DIM = 256  # Word embedding dimension
    TAGGER_CHAR_EMBED_DIM = 64  # Character embedding dimension
    TAGGER_CNN_FILTERS = 128  # Number of filters for Char CNN
    TAGGER_CNN_KERNEL_SIZE = 3  # Kernel size for Char CNN
    TAGGER_HIDDEN_DIM = 512  # LSTM hidden dimension
    TAGGER_NUM_LAYERS = 2  # Number of Bi-LSTM layers
    TAGGER_DROPOUT = 0.4  # Dropout rate

    # 2. Transformer Seq2Seq Fallback (Character-Level)
    SEQ2SEQ_D_MODEL = 256  # Transformer embedding dimension
    SEQ2SEQ_NHEAD = 8  # Number of attention heads
    SEQ2SEQ_NUM_ENCODER_LAYERS = 4  # Encoder depth
    SEQ2SEQ_NUM_DECODER_LAYERS = 4  # Decoder depth
    SEQ2SEQ_DIM_FEEDFORWARD = 1024  # FFN dimension
    SEQ2SEQ_DROPOUT = 0.2  # Dropout rate

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 256  # Batch size (A100 can handle large batches)
    LEARNING_RATE = 1e-3  # Initial learning rate
    WEIGHT_DECAY = 1e-5  # L2 regularization

    # Epochs
    TAGGER_EPOCHS = 10  # Bi-LSTMs converge relatively quickly
    SEQ2SEQ_EPOCHS = 20  # Transformers may need more time

    # Early Stopping & Scheduler
    PATIENCE = 3  # Epochs to wait before early stopping
    SCHEDULER_FACTOR = 0.5  # Factor to reduce LR by
    SCHEDULER_PATIENCE = 1  # Epochs to wait before reducing LR
