import os
import torch


class Config:
    # =========================================================================
    # GLOBAL SETTINGS
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # DIRECTORIES & PATHS
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_13"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Input Metadata
    TRAIN_DATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "en_sample_submission_2.csv")

    # Vocabularies & Tokenizers
    VOCAB_WORDS_PATH = os.path.join(WORKING_DIR, "vocab_words.json")
    VOCAB_CHARS_PATH = os.path.join(WORKING_DIR, "vocab_chars.json")
    VOCAB_CLASSES_PATH = os.path.join(WORKING_DIR, "vocab_classes.json")
    BPE_MODEL_PREFIX = os.path.join(WORKING_DIR, "bpe_tokenizer")

    # Cache Files (Parquet/NPY for speed)
    KNOWLEDGE_BASE_PATH = os.path.join(WORKING_DIR, "knowledge_base.parquet")
    PRIORS_PATH = os.path.join(WORKING_DIR, "priors.parquet")

    # Processed Data Cache
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.npy")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.npy")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.npy")

    # Model Checkpoints
    TAGGER_MODEL_PATH = os.path.join(WORKING_DIR, "tagger_best_model.pth")
    SEQ2SEQ_MODEL_PATH = os.path.join(WORKING_DIR, "seq2seq_best_model.pth")

    # Submission
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # DATA PROCESSING HYPERPARAMETERS
    # =========================================================================
    MAX_SEQ_LEN = 128  # Max tokens per sentence
    MAX_WORD_LEN = 30  # Max characters per token for Char-CNN
    VOCAB_SIZE_WORDS = 100000  # Max word vocabulary size
    VOCAB_SIZE_BPE = 32000  # BPE vocabulary size
    MIN_FREQ = 2  # Minimum frequency for a word to be in vocab

    # =========================================================================
    # REGEX FEATURES (Explicit Morphological Features)
    # =========================================================================
    # These patterns define the binary flags used in the input representation
    REGEX_PATTERNS = [
        r"^\d+$",  # Is all digits
        r"^\d+\.\d+$",  # Is decimal
        r"^\d{1,3}(,\d{3})+$",  # Has comma separators
        r"^\d{1,3}(,\d{3})*\.\d+$",  # Comma separated decimal
        r"^\d{4}$",  # 4 digits (potential year)
        r"^\d{1,2}/\d{1,2}/\d{2,4}$",  # Date slash format
        r"^\d{1,2}-\d{1,2}-\d{2,4}$",  # Date dash format
        r"^\d{1,2}:\d{2}$",  # Time format
        r"^\d{1,2}:\d{2}:\d{2}$",  # Time with seconds
        r"^[$£€¥]\d",  # Starts with currency symbol
        r"\d[%]$",  # Ends with percent
        r"^(http|https|www)",  # URL start
        r".+@.+\..+",  # Email format
        r"^[A-Z\.]+$",  # All caps/Acronym
        r"^[A-Z][a-z]+$",  # Title case
        r".*[0-9].*",  # Contains any digit
        r".*[-].*",  # Contains hyphen
        r".*[/].*",  # Contains slash
        r"#",  # Hashtag
        r"@",  # At symbol
    ]
    REGEX_DIM = len(REGEX_PATTERNS)

    # =========================================================================
    # MODEL ARCHITECTURE: TAGGER (Bi-LSTM)
    # =========================================================================
    # Input Dimensions
    WORD_EMBED_DIM = 128
    CHAR_EMBED_DIM = 32
    BPE_EMBED_DIM = 64
    PRIOR_DIM = 20  # Approx number of classes (PLAIN, PUNCT, DATE, etc.)

    # Char CNN
    CHAR_CNN_FILTERS = 32
    CHAR_CNN_KERNEL_SIZE = 3

    # Main Backbone
    LSTM_HIDDEN_DIM = 256
    LSTM_LAYERS = 2
    LSTM_DROPOUT = 0.3

    # Feature Dropout (Regularization for Priors/Regex)
    FEATURE_DROPOUT = 0.4

    # =========================================================================
    # MODEL ARCHITECTURE: FALLBACK (Seq2Seq LSTM)
    # =========================================================================
    SEQ2SEQ_EMBED_DIM = 64
    SEQ2SEQ_HIDDEN_DIM = 256
    SEQ2SEQ_LAYERS = 1
    SEQ2SEQ_DROPOUT = 0.2
    MAX_GEN_LEN = 128  # Max length for generated text

    # =========================================================================
    # TRAINING HYPERPARAMETERS
    # =========================================================================
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10
    PATIENCE = 3  # Early stopping patience
    GRAD_CLIP = 1.0  # Gradient clipping value

    # Scheduler
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 1

    # Loss Weights
    # We use square-root smoothing, calculated dynamically, but can set a default
    USE_CLASS_WEIGHTS = True
