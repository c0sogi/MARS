import os


class Config:
    """
    Centralized configuration for the Early-Fusion Convolutional Ranker
    and Dynamic Kernel Convolutional Reader pipeline.
    """

    # =========================================================================
    # File Paths and Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_10"
    SUBMISSION_DIR = "./submission"

    # Input Files
    TRAIN_FILE = os.path.join(INPUT_DIR, "simplified-nq-train.jsonl")
    # Note: Test file is usually simplified-nq-test.jsonl or simplified-nq-kaggle-test.jsonl
    # The pipeline should handle the glob/search, but here is a default
    TEST_FILE_PATTERN = os.path.join(INPUT_DIR, "*test.jsonl")
    SAMPLE_SUBMISSION_FILE = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache Files (Parquet/Numpy for deterministic processing)
    VOCAB_PATH = os.path.join(WORKING_DIR, "vocab.parquet")
    EMBEDDING_MATRIX_PATH = os.path.join(WORKING_DIR, "embedding_matrix.npy")

    # Processed Datasets for Models
    RANKER_TRAIN_DATA_PATH = os.path.join(WORKING_DIR, "ranker_train_data.parquet")
    RANKER_VAL_DATA_PATH = os.path.join(WORKING_DIR, "ranker_val_data.parquet")

    READER_TRAIN_DATA_PATH = os.path.join(WORKING_DIR, "reader_train_data.parquet")
    READER_VAL_DATA_PATH = os.path.join(WORKING_DIR, "reader_val_data.parquet")

    RANKER_TEST_FEATURES_PATH = os.path.join(
        WORKING_DIR, "ranker_test_features.parquet"
    )

    # Model Checkpoints
    RANKER_MODEL_PATH = os.path.join(WORKING_DIR, "ranker_best.pth")
    READER_MODEL_PATH = os.path.join(WORKING_DIR, "reader_best.pth")

    # Final Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Processing Parameters
    # =========================================================================
    SEED = 42

    # Controls dataset size for debugging/development.
    # Set to an integer (e.g., 5000) to limit samples, or None for full dataset.
    SAMPLE_SIZE = 10000

    VOCAB_SIZE = 20000
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"
    SEP_TOKEN = "<SEP>"

    # Sequence Lengths
    MAX_QUESTION_LEN = 32
    MAX_PARAGRAPH_LEN = 256
    # Ranker input = [Question; <SEP>; Paragraph]
    MAX_RANKER_SEQ_LEN = MAX_QUESTION_LEN + MAX_PARAGRAPH_LEN + 1

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    EMBEDDING_DIM = 100  # Dimension for word embeddings (e.g., GloVe)

    # Early-Fusion CNN Ranker
    RANKER_FILTERS = 128
    RANKER_KERNEL_SIZE = 3
    RANKER_HIDDEN_DIM = 128

    # Dynamic Kernel Reader
    READER_FILTERS = 128
    READER_KERNEL_SIZE = 3  # Base kernel size, dynamic weights will match this

    DROPOUT_RATE = 0.3

    # =========================================================================
    # Training Parameters
    # =========================================================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 5
    EARLY_STOPPING_PATIENCE = 2

    # Threshold for predicting "Null" / No Answer
    CONFIDENCE_THRESHOLD = 0.5

    @staticmethod
    def ensure_directories():
        """Creates necessary working and submission directories."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
