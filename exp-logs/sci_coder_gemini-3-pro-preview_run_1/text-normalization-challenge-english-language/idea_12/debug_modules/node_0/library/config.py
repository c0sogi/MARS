import os
import torch


class Config:
    # =========================================================================
    # Paths & Directories
    # =========================================================================
    # Input Metadata (Pre-split)
    METADATA_DIR = "./metadata"
    TRAIN_DATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory for Caching and Models
    WORKING_DIR = "./working/idea_12"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Vocabulary Caches
    VOCAB_WORDS_PATH = os.path.join(WORKING_DIR, "vocab_words.json")
    VOCAB_CHARS_PATH = os.path.join(WORKING_DIR, "vocab_chars.json")
    VOCAB_CLASSES_PATH = os.path.join(WORKING_DIR, "vocab_classes.json")

    # Feature Caches (for deterministic reloading)
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.npy")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.npy")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.npy")

    # Knowledge Base Cache
    KNOWLEDGE_BASE_PATH = os.path.join(WORKING_DIR, "knowledge_base.parquet")

    # Model Checkpoints
    TAGGER_MODEL_PATH = os.path.join(WORKING_DIR, "tagger_best_model.pth")
    SEQ2SEQ_MODEL_PATH = os.path.join(WORKING_DIR, "seq2seq_best_model.pth")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # General Configuration
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # =========================================================================
    # Data Processing Hyperparameters
    # =========================================================================
    MAX_VOCAB_SIZE = 100000  # Max words in vocabulary
    MIN_FREQ = 2  # Minimum frequency to include a word
    MAX_CHAR_VOCAB = 256  # Max unique characters
    MAX_SEQ_LEN = 128  # Max sequence length for Seq2Seq generation

    # =========================================================================
    # Feature Engineering (Regex Patterns)
    # =========================================================================
    # These regex patterns are used to generate binary feature vectors
    # that help the Tagger identify token types (e.g., DATE, CARDINAL, MONEY).
    REGEX_PATTERNS = [
        r"^\d+$",  # All digits
        r"^\d+\.\d+$",  # Float
        r"^\d+,\d+$",  # Number with comma
        r"\d",  # Has digit
        r"^[A-Za-z]+$",  # All letters
        r"^[A-Z]+$",  # All caps
        r"^[A-Z][a-z]+$",  # Title case
        r"^[a-z]+$",  # All lower
        r"[A-Z]",  # Has capital letter
        r"\.",  # Has dot
        r",",  # Has comma
        r"[\$£€¥¢]",  # Currency symbol
        r"%",  # Percent
        r"#",  # Hash
        r"-",  # Dash
        r"/",  # Slash
        r":",  # Colon
        r"^(http|https|www)",  # URL start
        r"@",  # Email symbol
        r"(st|nd|rd|th)$",  # Ordinal suffix
        r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",  # Months
        r"^(mon|tue|wed|thu|fri|sat|sun)",  # Days
        r"(am|pm|a\.m\.|p\.m\.)$",  # Time suffix
        r"^[IVXLCDM]+$",  # Roman numerals
        r"^(km|kg|m|cm|mm|ft|in|g|mg|l|ml)$",  # Common units
    ]

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================

    # Stage 1: Morphologically-Enhanced Bi-LSTM Tagger
    TAGGER_EMBED_DIM = 256  # Word embedding dimension
    TAGGER_CHAR_EMBED_DIM = 64  # Character embedding dimension
    TAGGER_CNN_FILTERS = 128  # Number of filters for Char-CNN
    TAGGER_CNN_KERNEL_SIZE = 3  # Kernel size for Char-CNN
    TAGGER_HIDDEN_DIM = 512  # LSTM hidden size (bidirectional = 2x this)
    TAGGER_LAYERS = 2  # Number of LSTM layers
    TAGGER_DROPOUT = 0.3  # Dropout rate

    # Stage 2: LSTM Seq2Seq Fallback
    SEQ2SEQ_EMBED_DIM = 128  # Character embedding dimension for Seq2Seq
    SEQ2SEQ_HIDDEN_DIM = 512  # LSTM hidden size
    SEQ2SEQ_LAYERS = 1  # Number of LSTM layers
    SEQ2SEQ_DROPOUT = 0.2  # Dropout rate
    TEACHER_FORCING_RATIO = 0.5  # Probability of using true target as next input

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 256  # Batch size (A100 can handle large batches)
    LEARNING_RATE = 1e-3  # Initial learning rate
    WEIGHT_DECAY = 1e-5  # L2 Regularization
    EPOCHS = 15  # Maximum number of epochs
    PATIENCE = 3  # Early stopping patience

    # Loss Function Configuration
    CLASS_WEIGHT_POWER = 0.5  # Power for Square-Root Smoothing (0.5 = sqrt)

    # Scheduler Configuration (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.5  # Factor to reduce LR by
    SCHEDULER_PATIENCE = 1  # Patience for scheduler

    # Debugging
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SIZE = 10000  # Number of samples for debug mode
