import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42

    # --------------------------------------------------------------------------
    # Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_13"
    SUBMISSION_DIR = "./submission"

    # --------------------------------------------------------------------------
    # Input Files
    # --------------------------------------------------------------------------
    TRAIN_FILE = os.path.join(INPUT_DIR, "simplified-nq-train.jsonl")
    TEST_FILE = os.path.join(INPUT_DIR, "simplified-nq-test.jsonl")
    SAMPLE_SUBMISSION_FILE = os.path.join(INPUT_DIR, "sample_submission.csv")

    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # --------------------------------------------------------------------------
    # Caching Paths (Idea 13 Specific)
    # --------------------------------------------------------------------------
    # Vocabulary and Embeddings
    VOCAB_PATH = os.path.join(WORKING_DIR, "vocab.parquet")
    EMBEDDING_MATRIX_PATH = os.path.join(WORKING_DIR, "embedding_matrix.npy")

    # Ranker Data
    RANKER_TRAIN_DATA = os.path.join(WORKING_DIR, "ranker_train_data.parquet")
    RANKER_VAL_DATA = os.path.join(WORKING_DIR, "ranker_val_data.parquet")
    RANKER_TEST_DATA = os.path.join(WORKING_DIR, "ranker_test_data.parquet")

    # Reader Data
    READER_TRAIN_DATA = os.path.join(WORKING_DIR, "reader_train_data.parquet")
    READER_VAL_DATA = os.path.join(WORKING_DIR, "reader_val_data.parquet")
    READER_TEST_DATA = os.path.join(WORKING_DIR, "reader_test_data.parquet")

    # Model Checkpoints
    RANKER_MODEL_PATH = os.path.join(WORKING_DIR, "ranker_best.pth")
    READER_MODEL_PATH = os.path.join(WORKING_DIR, "reader_best.pth")

    # Final Output
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Processing Hyperparameters
    # --------------------------------------------------------------------------
    MAX_Q_LEN = 30  # Maximum tokens for Question
    MAX_DOC_LEN = 300  # Maximum tokens for Candidate Paragraph
    VOCAB_SIZE = 40000  # Maximum vocabulary size
    EMBEDDING_DIM = 100  # Dimension of pre-trained embeddings (e.g., GloVe)
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"

    # --------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # --------------------------------------------------------------------------
    # Ranker (Direct Interaction Pooling Network)
    RANKER_HIDDEN_DIM = 128
    RANKER_DROPOUT = 0.3

    # Reader (Query-Initialized Recurrent Network)
    READER_HIDDEN_DIM = 128  # LSTM Hidden Dimension
    READER_DROPOUT = 0.3

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10
    EARLY_STOPPING_PATIENCE = 2

    # --------------------------------------------------------------------------
    # Inference Hyperparameters
    # --------------------------------------------------------------------------
    CONFIDENCE_THRESHOLD = 0.4  # Threshold for predicting a non-null answer

    # --------------------------------------------------------------------------
    # Execution Control
    # --------------------------------------------------------------------------
    # Set DEBUG to True to run on a small subset of data for testing the pipeline
    DEBUG = False

    # Number of samples to use if DEBUG is True. If None, use full dataset.
    # If DEBUG is False, this should be None or a very large number.
    SAMPLE_SIZE = 5000 if DEBUG else None

    # Hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def setup():
        """Ensures necessary directories exist."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        print(f"Configured for device: {Config.DEVICE}")
        print(f"Working Directory: {Config.WORKING_DIR}")
        print(f"Debug Mode: {Config.DEBUG} (Sample Size: {Config.SAMPLE_SIZE})")
