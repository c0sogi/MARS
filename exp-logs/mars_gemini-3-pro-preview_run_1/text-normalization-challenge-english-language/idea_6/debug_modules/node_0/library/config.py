import os
import torch


class Config:
    """
    Configuration for Text Normalization Task.
    Includes paths, hyperparameters for Tagger (Bi-LSTM-CRF) and Seq2Seq (Transformer),
    and training settings.
    """

    # --------------------------------------------------------------------------
    # General System Settings
    # --------------------------------------------------------------------------
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # Debugging / Development
    DEBUG = False  # Set to True to use a smaller subset of data
    DEBUG_SIZE = 50000  # Number of sentences to use in debug mode

    # --------------------------------------------------------------------------
    # File Paths
    # --------------------------------------------------------------------------
    # Input Metadata (Read-Only)
    METADATA_DIR = "./metadata"
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.csv")
    VAL_FILE = os.path.join(METADATA_DIR, "val.csv")
    TEST_FILE = os.path.join(METADATA_DIR, "test.csv")

    # Raw Input (for submission format reference)
    SAMPLE_SUBMISSION = "./input/en_sample_submission_2.csv"

    # Working Directory (Read/Write)
    # Stores intermediate artifacts like vocabularies and model checkpoints
    WORK_DIR = "./working/idea_6"

    # Data Artifacts (Parquet format)
    VOCAB_TOKENS_PATH = os.path.join(WORK_DIR, "vocab_tokens.parquet")
    VOCAB_CLASSES_PATH = os.path.join(WORK_DIR, "vocab_classes.parquet")
    VOCAB_CHARS_PATH = os.path.join(WORK_DIR, "vocab_chars.parquet")
    KNOWLEDGE_BASE_PATH = os.path.join(WORK_DIR, "knowledge_base.parquet")

    # Model Checkpoints
    TAGGER_MODEL_PATH = os.path.join(WORK_DIR, "tagger_best_model.pth")
    SEQ2SEQ_MODEL_PATH = os.path.join(WORK_DIR, "seq2seq_best_model.pth")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Processing Parameters
    # --------------------------------------------------------------------------
    MAX_VOCAB_SIZE = 100000  # Max words in token vocabulary
    MIN_FREQ = 2  # Minimum frequency for a token to be included

    # Sequence Lengths (Based on EDA: Max tokens/sent=233, Max chars/token=1057)
    MAX_SEQ_LEN = 256  # Max tokens per sentence (padding/truncating limit)
    MAX_CHAR_LEN = 50  # Max characters per token (for Char CNN & Seq2Seq)

    # Special Tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"

    # --------------------------------------------------------------------------
    # Stage 1: Tagger Model Hyperparameters (Bi-LSTM-CRF)
    # --------------------------------------------------------------------------
    TAGGER_EMBEDDING_DIM = 300
    TAGGER_HIDDEN_DIM = 512
    TAGGER_NUM_LAYERS = 2
    TAGGER_DROPOUT = 0.3
    TAGGER_USE_CRF = True  # Use Conditional Random Field for global consistency

    # Character-level features for Tagger (CNN)
    TAGGER_USE_CHAR_CNN = True
    TAGGER_CHAR_EMBEDDING_DIM = 50
    TAGGER_CHAR_CNN_FILTERS = 50
    TAGGER_CHAR_CNN_KERNEL_SIZE = 3

    # --------------------------------------------------------------------------
    # Stage 2: Fallback Model Hyperparameters (Transformer Seq2Seq)
    # --------------------------------------------------------------------------
    # Input: Character sequence of token + Class ID embedding
    SEQ2SEQ_EMBEDDING_DIM = 128
    SEQ2SEQ_HIDDEN_DIM = 512  # FeedForward dimension
    SEQ2SEQ_NUM_LAYERS = 3  # Number of Encoder and Decoder layers
    SEQ2SEQ_NUM_HEADS = 4
    SEQ2SEQ_DROPOUT = 0.1
    SEQ2SEQ_MAX_OUTPUT_LEN = 128  # Max length for generated normalized text

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 128
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5

    # Optimization
    PATIENCE = 3  # Early stopping patience
    GRAD_CLIP = 1.0  # Gradient clipping threshold

    # Loss Weighting
    USE_CLASS_WEIGHTS = True
    CLASS_WEIGHT_SMOOTHING = 0.5  # Power for sqrt smoothing (0.5 = sqrt(N/Nc))

    @classmethod
    def setup(cls):
        """
        Ensures that the working and submission directories exist.
        Should be called at the start of execution.
        """
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Configuration Setup Complete.")
        print(f"Working Directory: {cls.WORK_DIR}")
        print(f"Submission Directory: {cls.SUBMISSION_DIR}")
        print(f"Device: {cls.DEVICE}")
