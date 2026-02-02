import os
import torch
import random
import numpy as np


class ProjectConfig:
    """
    Global project configuration for paths and environment settings.
    """

    # Base Directories
    BASE_DIR = "./working/idea_4"
    METADATA_DIR = "./metadata"
    INPUT_DIR = "./input"
    SUBMISSION_DIR = "./submission"

    # Ensure directories exist
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data Paths
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "en_sample_submission_2.csv")

    # Artifact Paths (Saved in working dir)
    VOCAB_WORDS_PATH = os.path.join(BASE_DIR, "vocab_words.parquet")
    VOCAB_CHARS_PATH = os.path.join(BASE_DIR, "vocab_chars.parquet")
    VOCAB_CLASSES_PATH = os.path.join(BASE_DIR, "vocab_classes.parquet")
    KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "knowledge_base.parquet")

    TAGGER_MODEL_PATH = os.path.join(BASE_DIR, "tagger_best_model.pth")
    SEQ2SEQ_MODEL_PATH = os.path.join(BASE_DIR, "seq2seq_best_model.pth")

    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Debug Mode (Set to True to run on a small subset for testing pipeline)
    DEBUG = False
    DEBUG_SIZE = 10000  # Number of sentences to use in debug mode


class DataConfig:
    """
    Configuration for data processing, vocabulary, and sequence constraints.
    """

    # Vocabulary Limits
    MAX_WORD_VOCAB_SIZE = (
        60000  # Large enough for core words, others handled by char features
    )
    MIN_WORD_FREQ = 2

    # Special Tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SOS_TOKEN = "<SOS>"  # Start of Sequence (for Seq2Seq)
    EOS_TOKEN = "<EOS>"  # End of Sequence (for Seq2Seq)

    # Sequence Lengths
    MAX_SENT_LEN = 128  # Max tokens per sentence for Tagger (covers >99% of data)
    MAX_TOKEN_LEN = 50  # Max characters per token for CNN/Seq2Seq

    # Data Loading
    NUM_WORKERS = 4
    PIN_MEMORY = True if torch.cuda.is_available() else False


class ModelConfig:
    """
    Hyperparameters for the Multi-Granularity Tagger and Seq2Seq Fallback models.
    """

    # --- Tagger (Bi-LSTM + Char CNN) ---
    TAGGER_WORD_EMBED_DIM = 300
    TAGGER_CHAR_EMBED_DIM = 50

    # Char CNN
    TAGGER_CNN_FILTERS = 64
    TAGGER_CNN_KERNEL_SIZE = 3

    # Bi-LSTM Backbone
    TAGGER_LSTM_HIDDEN_DIM = 256
    TAGGER_LSTM_LAYERS = 2
    TAGGER_DROPOUT = 0.3
    TAGGER_BIDIRECTIONAL = True

    # --- Seq2Seq Fallback (Char-level Encoder-Decoder) ---
    SEQ_EMBED_DIM = 128
    SEQ_HIDDEN_DIM = 256
    SEQ_LAYERS = 1
    SEQ_DROPOUT = 0.2
    TEACHER_FORCING_RATIO = 0.5


class TrainingConfig:
    """
    Training hyperparameters and optimization settings.
    """

    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Tagger Training
    TAGGER_BATCH_SIZE = 64
    TAGGER_EPOCHS = 10
    TAGGER_LR = 1e-3
    TAGGER_WEIGHT_DECAY = 1e-5
    TAGGER_GRAD_CLIP = 1.0

    # Seq2Seq Training
    SEQ_BATCH_SIZE = 128
    SEQ_EPOCHS = 15
    SEQ_LR = 1e-3
    SEQ_WEIGHT_DECAY = 1e-5
    SEQ_GRAD_CLIP = 1.0

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_PATIENCE = 2
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_MIN_LR = 1e-6

    # Class Weighting
    # Smoothing factor for Square-Root Smoothed Class Weights
    # Weights = sqrt(Total / Count)
    USE_CLASS_WEIGHTS = True


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
