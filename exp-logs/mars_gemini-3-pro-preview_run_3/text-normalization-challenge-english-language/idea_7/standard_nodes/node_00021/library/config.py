import os
import torch
import string


class Config:
    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # File Paths
    # ==========================================
    # Input Metadata (Read-Only)
    TRAIN_META_PATH = "./metadata/train.parquet"
    VAL_META_PATH = "./metadata/val.parquet"
    TEST_META_PATH = "./metadata/test.parquet"

    # Working Directory (Write Access)
    WORK_DIR = "./working/idea_7"

    # Cache Files (Parquet/PT)
    STATS_CACHE_DIR = os.path.join(WORK_DIR, "stats")
    PROCESSED_DATA_DIR = os.path.join(WORK_DIR, "processed")
    MODEL_CHECKPOINT_PATH = os.path.join(WORK_DIR, "model_best.pt")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Ensure directories exist
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(STATS_CACHE_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    MAX_LEN = 128  # Maximum sequence length for character sequences
    CONTEXT_WINDOW = 1  # Number of tokens to look ahead/behind (1 = Trigram context)

    # Soft Filtering: Probability of including PLAIN/PUNCT tokens in neural training
    # to prevent training-inference skew.
    SOFT_FILTER_RATIO = 0.05

    # ==========================================
    # Model Architecture Hyperparameters
    # ==========================================
    # Factored Embeddings
    CHAR_EMB_DIM = 64
    CASE_EMB_DIM = 8
    TYPE_EMB_DIM = 8

    # RNN Backbone
    ENC_HIDDEN_DIM = 256
    DEC_HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.2

    # Attention
    ATTN_DIM = 128

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 15
    PATIENCE = 3  # Early stopping patience
    TEACHER_FORCING_RATIO = 0.5
    CLIP_GRAD = 1.0

    # Multi-Task Loss Weight
    # Total Loss = L_gen + LAMBDA_AUX * L_class
    LAMBDA_AUX = 0.5

    # ==========================================
    # Vocabulary and Special Tokens
    # ==========================================
    # Special Tokens
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    SEP_TOKEN = "<SEP>"  # Separator for context: prev <SEP> target <SEP> next
    UNK_TOKEN = "<UNK>"

    # Indices
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    SEP_IDX = 3
    UNK_IDX = 4

    # Base Character Vocabulary (Printable ASCII + Special)
    # We will build the final char2idx map dynamically or use this fixed set
    base_chars = list(string.printable)
    specials = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, SEP_TOKEN, UNK_TOKEN]
    CHAR_VOCAB = specials + base_chars
    VOCAB_SIZE = len(CHAR_VOCAB)

    # Factored Embedding Enums/Maps
    # Case Factors
    CASE_TYPES = ["<PAD>", "NONE", "UPPER", "LOWER"]
    CASE_VOCAB_SIZE = len(CASE_TYPES)

    # Character Type Factors
    TYPE_TYPES = ["<PAD>", "OTHER", "DIGIT", "LETTER", "SYMBOL"]
    TYPE_VOCAB_SIZE = len(TYPE_TYPES)

    # Target Classes (for Auxiliary Head)
    # Based on standard Text Normalization datasets (Google Text Norm)
    CLASSES = [
        "PLAIN",
        "PUNCT",
        "DATE",
        "LETTERS",
        "CARDINAL",
        "VERBATIM",
        "DECIMAL",
        "MEASURE",
        "MONEY",
        "ORDINAL",
        "TIME",
        "ELECTRONIC",
        "DIGIT",
        "FRACTION",
        "TELEPHONE",
        "ADDRESS",
    ]
    NUM_CLASSES = len(CLASSES)

    # ==========================================
    # Compute
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_char_map():
        """Returns dictionary mapping characters to indices."""
        return {c: i for i, c in enumerate(Config.CHAR_VOCAB)}

    @staticmethod
    def get_class_map():
        """Returns dictionary mapping class names to indices."""
        return {c: i for i, c in enumerate(Config.CLASSES)}
