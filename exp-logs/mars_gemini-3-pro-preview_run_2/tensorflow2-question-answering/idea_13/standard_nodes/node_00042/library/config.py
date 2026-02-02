import os


class Config:
    """
    Configuration for the Window-Based Max-Pooling Network for Natural Questions.
    """

    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_13"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"

    # Raw Data
    TRAIN_DATA_PATH = os.path.join(INPUT_DIR, "simplified-nq-train.jsonl")
    TEST_DATA_PATH = os.path.join(INPUT_DIR, "simplified-nq-test.jsonl")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache Files
    VOCAB_PATH = os.path.join(CACHE_DIR, "vocab.npy")
    EMBEDDING_MATRIX_PATH = os.path.join(CACHE_DIR, "embedding_matrix.npy")

    # Model Checkpoints
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    FINAL_SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Data Processing Parameters ---
    # Windowing logic: Split long answer candidates into overlapping windows
    WINDOW_SIZE = 128  # Number of tokens per window
    WINDOW_STRIDE = 64  # Stride for sliding window (overlap)
    MAX_WINDOWS_PER_CANDIDATE = 8  # Max windows to process per candidate to save memory

    # Sequence lengths
    MAX_QUESTION_LEN = 32  # Max tokens for the question

    # Vocabulary & Embeddings
    VOCAB_SIZE = 40000  # Maximum size of vocabulary
    EMBEDDING_DIM = 100  # Dimension of word embeddings
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"

    # --- Model Architecture ---
    HIDDEN_DIM = 256  # Hidden dimension for MLP layers
    DROPOUT = 0.3  # Dropout rate
    NUM_YES_NO_CLASSES = 3  # YES, NO, NONE

    # --- Training Hyperparameters ---
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 8
    EARLY_STOPPING_PATIENCE = 2

    # Negative Sampling: How many negative windows to include per positive window in training
    NEGATIVE_SAMPLING_RATIO = 3.0

    # Loss Weights
    LOSS_WEIGHT_WINDOW = 1.0  # Binary classification (Is this window relevant?)
    LOSS_WEIGHT_SPAN = 1.0  # Start/End token prediction
    LOSS_WEIGHT_YESNO = 0.5  # Yes/No classification

    # --- Inference Parameters ---
    # Score threshold for a long answer to be considered a valid prediction
    # If max score < threshold, predict BLANK
    LONG_ANSWER_CONFIDENCE_THRESHOLD = 0.5

    # --- Reproducibility & Debugging ---
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data
    DEBUG_SIZE = 2000  # Number of samples to use in debug mode

    @classmethod
    def setup(cls):
        """Ensures necessary directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
