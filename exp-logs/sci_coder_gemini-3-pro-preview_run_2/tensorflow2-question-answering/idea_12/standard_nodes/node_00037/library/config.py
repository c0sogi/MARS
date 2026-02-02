import os


class Config:
    """
    Configuration class for the Modulated Feature-Selection Network (FiLM) pipeline.
    Centralizes hyperparameters, file paths, and model settings.
    """

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Directory & File Paths
    # -------------------------------------------------------------------------
    # Input Data (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_FILE = os.path.join(INPUT_DIR, "simplified-nq-train.jsonl")
    TEST_FILE = os.path.join(INPUT_DIR, "simplified-nq-test.jsonl")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working Directory (Writeable)
    WORKING_DIR = "./working/idea_12"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Model Checkpoints
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Files (Deterministic Data Processing)
    VOCAB_FILE = os.path.join(CACHE_DIR, "vocab.npy")
    EMBEDDING_MATRIX_FILE = os.path.join(CACHE_DIR, "embedding_matrix.npy")
    # Using parquet for large datasets as requested
    TRAIN_CACHE = os.path.join(CACHE_DIR, "train_features.parquet")
    VAL_CACHE = os.path.join(CACHE_DIR, "val_features.parquet")
    TEST_CACHE = os.path.join(CACHE_DIR, "test_features.parquet")

    # -------------------------------------------------------------------------
    # Data Preprocessing Hyperparameters
    # -------------------------------------------------------------------------
    MAX_Q_LEN = 20  # Fixed length for Question sequences
    MAX_CTX_LEN = 300  # Fixed length for Long Answer Candidate sequences

    # Vocabulary
    VOCAB_SIZE = 40000  # Maximum number of unique tokens to keep
    EMBEDDING_DIM = 100  # Dimension of pre-trained embeddings (e.g., GloVe)
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"

    # Sampling for Imbalance
    NEGATIVE_SAMPLE_RATIO = (
        1  # Ratio of negative candidates to positive ones per question in training
    )

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # -------------------------------------------------------------------------
    # CNN Encoders
    CNN_FILTERS = 128  # Number of filters for Question and Candidate CNNs
    CNN_KERNEL_SIZE = 3  # Kernel size for 1D convolution

    # FiLM (Feature-wise Linear Modulation)
    FILM_DIM = 128  # Dimension of modulation vectors (must match CNN_FILTERS)

    # Regularization
    DROPOUT = 0.3  # Dropout rate applied after pooling/modulation

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 5
    EARLY_STOPPING_PATIENCE = 2  # Stop if validation loss doesn't improve for 2 epochs

    # Loss Weights (Multi-Task Learning)
    LOSS_WEIGHT_RANKING = 1.0  # Weight for Long Answer Ranking (Binary CE)
    LOSS_WEIGHT_SPAN = 1.0  # Weight for Span Start/End (Sparse Categorical CE)
    LOSS_WEIGHT_YESNO = 0.5  # Weight for Yes/No Classification (Categorical CE)

    # -------------------------------------------------------------------------
    # Inference Hyperparameters
    # -------------------------------------------------------------------------
    LONG_ANSWER_THRESHOLD = (
        0.5  # Confidence threshold to predict a long answer vs BLANK
    )

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    @classmethod
    def setup(cls):
        """
        Creates necessary directories for working, caching, and submission.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
