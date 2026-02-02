import os


class Config:
    # -------------------------------------------------------------------------
    # Directory Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # -------------------------------------------------------------------------
    # Input File Paths
    # -------------------------------------------------------------------------
    TRAIN_DATA_FILE = os.path.join(INPUT_DIR, "simplified-nq-train.jsonl")
    TEST_DATA_FILE = os.path.join(INPUT_DIR, "simplified-nq-test.jsonl")
    SAMPLE_SUBMISSION_FILE = os.path.join(INPUT_DIR, "sample_submission.csv")

    # -------------------------------------------------------------------------
    # Metadata File Paths (Parquet)
    # -------------------------------------------------------------------------
    TRAIN_META_FILE = os.path.join(METADATA_DIR, "train.parquet")
    VAL_META_FILE = os.path.join(METADATA_DIR, "val.parquet")
    TEST_META_FILE = os.path.join(METADATA_DIR, "test.parquet")

    # -------------------------------------------------------------------------
    # Caching Paths (Processed Data & Artifacts)
    # -------------------------------------------------------------------------
    # Vocabulary and Embeddings
    VOCAB_CACHE_FILE = os.path.join(WORKING_DIR, "vocab.npy")
    EMBEDDING_MATRIX_CACHE_FILE = os.path.join(WORKING_DIR, "embedding_matrix.npy")

    # Processed Datasets (Features)
    TRAIN_FEATURES_CACHE = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_CACHE = os.path.join(WORKING_DIR, "val_features.parquet")
    SHORT_ANSWER_DATA_CACHE = os.path.join(WORKING_DIR, "short_answer_data.parquet")

    # Model Checkpoints
    LONG_ANSWER_MODEL_PATH = os.path.join(WORKING_DIR, "de_convnet_model.pth")
    SHORT_ANSWER_WEIGHTS_PATH = os.path.join(WORKING_DIR, "short_answer_weights.npy")

    # Submission Output
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Processing Hyperparameters
    # -------------------------------------------------------------------------
    MAX_SEQ_LEN = 300  # Max tokens for Long Answer Candidate
    MAX_QUES_LEN = 30  # Max tokens for Question
    VOCAB_SIZE = 20000  # Maximum vocabulary size
    EMBEDDING_DIM = 100  # Dimension of word embeddings
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"

    # Short Answer Sliding Window
    WINDOW_SIZE = 10  # Size of sliding window in tokens
    WINDOW_STRIDE = 5  # Stride for sliding window

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters (DE-ConvNet)
    # -------------------------------------------------------------------------
    CNN_KERNEL_SIZES = [3, 4, 5]  # N-gram sizes to capture
    CNN_NUM_FILTERS = 64  # Filters per kernel size
    DROPOUT_RATE = 0.2  # Dropout probability
    HIDDEN_DIM = 128  # Dimension of dense layer after concatenation

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5
    PATIENCE = 2  # Early stopping patience (epochs)

    # Class Imbalance Handling
    NEG_SAMPLE_RATIO = 0.2  # Ratio of negative long answer candidates to keep

    # Debugging / Development
    # Set to an integer (e.g., 1000) to limit dataset size for fast iteration.
    # Set to None for full training.
    TRAIN_SAMPLE_SIZE = None

    # -------------------------------------------------------------------------
    # Inference Thresholds
    # -------------------------------------------------------------------------
    LONG_ANSWER_THRESHOLD = 0.5  # Alpha: Probability threshold for Long Answer
    SHORT_ANSWER_THRESHOLD = 0.5  # Beta: Score threshold for Short Answer

    @classmethod
    def setup(cls):
        """
        Creates necessary writable directories for caching and submission.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
