import os
import torch


class Config:
    """
    Configuration for the Text Normalization Project.
    Implements the 'Prior-Informed Multi-Feature Bi-LSTM with Inductive-Bias-Aligned Fallback' strategy.
    """

    # --------------------------------------------------------------------------
    # 1. General Configuration & Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SIZE = 5000  # Number of samples to use in debug mode
    NUM_WORKERS = 4  # Number of dataloader workers

    # --------------------------------------------------------------------------
    # 2. Directory Structure & Paths
    # --------------------------------------------------------------------------
    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_14"
    SUBMISSION_DIR = "./submission"

    # Create necessary writable directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Data Files (Metadata)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "en_sample_submission_2.csv")

    # --------------------------------------------------------------------------
    # 3. Caching & Artifacts
    # --------------------------------------------------------------------------
    # Tokenizer & Vocabularies
    BPE_MODEL_PREFIX = os.path.join(
        WORKING_DIR, "bpe_tokenizer"
    )  # SentencePiece adds .model/.vocab
    VOCAB_WORDS_PATH = os.path.join(WORKING_DIR, "vocab_words.json")
    VOCAB_CHARS_PATH = os.path.join(WORKING_DIR, "vocab_chars.json")
    VOCAB_CLASSES_PATH = os.path.join(WORKING_DIR, "vocab_classes.json")

    # Knowledge Base & Priors
    KNOWLEDGE_BASE_PATH = os.path.join(WORKING_DIR, "knowledge_base.parquet")
    PRIORS_PATH = os.path.join(WORKING_DIR, "priors.parquet")

    # Processed Data Cache (PyTorch Tensors / Objects)
    # Tagger: Sentence-level features
    TAGGER_TRAIN_DATA = os.path.join(WORKING_DIR, "tagger_train_data.pt")
    TAGGER_VAL_DATA = os.path.join(WORKING_DIR, "tagger_val_data.pt")

    # Seq2Seq: Token-level pairs (before, after) for changed tokens
    SEQ2SEQ_TRAIN_DATA = os.path.join(WORKING_DIR, "seq2seq_train_data.pt")
    SEQ2SEQ_VAL_DATA = os.path.join(WORKING_DIR, "seq2seq_val_data.pt")

    # Model Checkpoints
    TAGGER_MODEL_PATH = os.path.join(WORKING_DIR, "tagger_best_model.pth")
    SEQ2SEQ_MODEL_PATH = os.path.join(WORKING_DIR, "seq2seq_best_model.pth")

    # Final Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # 4. Data Processing Parameters
    # --------------------------------------------------------------------------
    # Vocabulary Limits
    MAX_WORD_VOCAB_SIZE = 100000
    BPE_VOCAB_SIZE = 10000
    MIN_WORD_FREQ = 2

    # Sequence Lengths
    MAX_SENT_LEN = 128  # Max tokens per sentence (for Tagger padding)
    MAX_TOKEN_LEN = 30  # Max chars per token (for Char-CNN and Seq2Seq)
    SEQ2SEQ_MAX_OUTPUT_LEN = 100  # Max length of generated normalized text

    # Special Tokens
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"

    # Regex Features (Explicit Morphological Cues)
    # These patterns provide the Tagger with explicit signals for rare classes
    REGEX_PATTERNS = [
        r"^\d+$",  # Digits only
        r"^\d+\.\d+$",  # Decimal
        r"^\d+,\d+$",  # Digit with comma
        r"^[A-Z]+$",  # All caps
        r"^[A-Z][a-z]+$",  # Title case
        r"^\d{4}$",  # 4 Digits (Year-like)
        r"^\d{1,2}:\d{2}$",  # Time format
        r"^[\$€£¥]",  # Currency symbol start
        r"%$",  # Percent symbol end
        r"^\d+(st|nd|rd|th)$",  # Ordinal
        r"^[a-zA-Z]$",  # Single letter
        r"http|www|\.com",  # URL parts
        r"#",  # Hashtag
        r"@",  # Mention
        r"[-/]",  # Dash or Slash (Dates/Fractions)
    ]
    NUM_REGEX_FEATURES = len(REGEX_PATTERNS)

    # --------------------------------------------------------------------------
    # 5. Model Hyperparameters
    # --------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Stage 1: Prior-Informed Bi-LSTM Tagger ---
    TAGGER_EMBED_DIM = 256  # Word embedding dimension
    TAGGER_BPE_EMBED_DIM = 128  # BPE embedding dimension
    TAGGER_CHAR_EMBED_DIM = 64  # Character embedding dimension
    TAGGER_CHAR_CNN_FILTERS = 64  # Number of filters for Char-CNN
    TAGGER_CHAR_CNN_KERNEL_SIZE = 3  # Kernel size for Char-CNN
    TAGGER_HIDDEN_DIM = 512  # LSTM hidden state dimension
    TAGGER_NUM_LAYERS = 2  # Number of Bi-LSTM layers
    TAGGER_DROPOUT = 0.3  # General dropout
    TAGGER_FEATURE_DROPOUT = (
        0.4  # Dropout specifically for Priors/Regex to force context learning
    )

    # --- Stage 2: LSTM Seq2Seq Fallback ---
    SEQ2SEQ_EMBED_DIM = 128
    SEQ2SEQ_HIDDEN_DIM = 512
    SEQ2SEQ_NUM_LAYERS = 2
    SEQ2SEQ_DROPOUT = 0.2
    SEQ2SEQ_ATTENTION = True  # Enable attention mechanism

    # --------------------------------------------------------------------------
    # 6. Training Hyperparameters
    # --------------------------------------------------------------------------
    # Tagger Training
    TAGGER_BATCH_SIZE = 64  # Sentences per batch
    TAGGER_LR = 1e-3
    TAGGER_WEIGHT_DECAY = 1e-5
    TAGGER_EPOCHS = 10
    TAGGER_PATIENCE = 3  # Early stopping patience
    USE_CLASS_WEIGHTS = True  # Use sqrt(N/Nc) class weights

    # Seq2Seq Training
    SEQ2SEQ_BATCH_SIZE = 256  # Token pairs per batch
    SEQ2SEQ_LR = 1e-3
    SEQ2SEQ_WEIGHT_DECAY = 1e-5
    SEQ2SEQ_EPOCHS = 15
    SEQ2SEQ_PATIENCE = 3
    TEACHER_FORCING_RATIO = 0.5
