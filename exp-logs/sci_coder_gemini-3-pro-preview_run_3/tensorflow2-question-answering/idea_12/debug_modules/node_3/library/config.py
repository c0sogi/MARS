import os


class Config:
    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Directory Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_12"
    SUBMISSION_DIR = "./submission"

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    # Raw Data
    TRAIN_RAW_FILE = os.path.join(INPUT_DIR, "simplified-nq-train.jsonl")
    TEST_RAW_FILE = os.path.join(INPUT_DIR, "simplified-nq-test.jsonl")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cached Artifacts (Created during processing)
    VOCAB_PATH = os.path.join(WORKING_DIR, "vocab.parquet")
    EMBEDDING_MATRIX_PATH = os.path.join(WORKING_DIR, "embedding_matrix.npy")

    # Processed Datasets
    RANKER_TRAIN_PATH = os.path.join(WORKING_DIR, "ranker_train_data.parquet")
    RANKER_VAL_PATH = os.path.join(WORKING_DIR, "ranker_val_data.parquet")
    RANKER_TEST_PATH = os.path.join(WORKING_DIR, "ranker_test_features.parquet")

    READER_TRAIN_PATH = os.path.join(WORKING_DIR, "reader_train_data.parquet")
    READER_VAL_PATH = os.path.join(WORKING_DIR, "reader_val_data.parquet")
    READER_TEST_PATH = os.path.join(WORKING_DIR, "reader_test_features.parquet")

    # Model Checkpoints
    RANKER_MODEL_PATH = os.path.join(WORKING_DIR, "ranker_best.pth")
    READER_MODEL_PATH = os.path.join(WORKING_DIR, "reader_best.pth")

    # Output
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Processing Parameters
    # -------------------------------------------------------------------------
    VOCAB_SIZE = 40000  # Maximum vocabulary size
    EMBEDDING_DIM = 100  # Dimension for word embeddings (e.g., GloVe)

    # Sequence Lengths
    Q_MAX_LEN = 32  # Maximum length for Question tokens
    P_MAX_LEN = 256  # Maximum length for Paragraph/Candidate tokens

    # Candidate Generation
    MAX_CANDIDATES_PER_DOC = 25  # Maximum number of paragraphs to extract per document
    NEG_RATIO = 2  # Negative paragraphs per positive paragraph for Ranker training

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # -------------------------------------------------------------------------
    # Ranker: 2D Interaction Convolutional Network
    RANKER_CONV_FILTERS = [32, 64]  # Filters for consecutive 2D Conv layers
    RANKER_KERNEL_SIZES = [3, 3]  # Kernel sizes for 2D Conv layers
    RANKER_POOL_SIZES = [2, 2]  # Pooling sizes
    RANKER_HIDDEN_DIM = 128  # Dense layer dimension
    RANKER_DROPOUT = 0.3

    # Reader: 1D U-Net
    # Encoder (Contraction Path) filters
    READER_ENC_FILTERS = [64, 128]
    # Decoder (Expansion Path) filters
    READER_DEC_FILTERS = [128, 64]
    READER_KERNEL_SIZE = 3
    READER_DROPOUT = 0.3

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10
    EARLY_STOPPING_PATIENCE = 3

    # Dataset Subsampling
    # Set to an integer (e.g., 10000) for debugging or limited resource training.
    # Set to None to use the full dataset provided in metadata.
    TRAIN_SAMPLE_SIZE = 50000
    VAL_SAMPLE_SIZE = 5000

    # -------------------------------------------------------------------------
    # Inference / Post-processing
    # -------------------------------------------------------------------------
    # Minimum score for a long answer to be considered valid
    LONG_ANSWER_THRESHOLD = 0.5
    # Minimum joint probability (start_prob * end_prob) for a short answer
    SHORT_ANSWER_THRESHOLD = 0.1

    @staticmethod
    def setup_directories():
        """Ensures that the necessary working and submission directories exist."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
