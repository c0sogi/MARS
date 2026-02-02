import os


class Config:
    # --------------------------------------------------------------------------
    # Directory and File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_8"
    SUBMISSION_DIR = "./submission"

    # Input Files
    TRAIN_FILE = os.path.join(INPUT_DIR, "simplified-nq-train.jsonl")
    TEST_FILE_PATTERN = os.path.join(INPUT_DIR, "*test.jsonl")  # Glob pattern for test
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache Files (Parquet/Numpy)
    VOCAB_CACHE_PATH = os.path.join(WORKING_DIR, "vocab.parquet")
    EMBEDDING_MATRIX_PATH = os.path.join(WORKING_DIR, "embedding_matrix.npy")

    # Processed Data Cache Paths
    RANKER_TRAIN_DATA_PATH = os.path.join(WORKING_DIR, "ranker_train_data.parquet")
    RANKER_VAL_DATA_PATH = os.path.join(WORKING_DIR, "ranker_val_data.parquet")
    READER_TRAIN_DATA_PATH = os.path.join(WORKING_DIR, "reader_train_data.parquet")
    READER_VAL_DATA_PATH = os.path.join(WORKING_DIR, "reader_val_data.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # Model Checkpoints
    RANKER_MODEL_PATH = os.path.join(WORKING_DIR, "ranker_best.pth")
    READER_MODEL_PATH = os.path.join(WORKING_DIR, "reader_best.pth")

    # Final Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Global Settings
    # --------------------------------------------------------------------------
    SEED = 42
    NUM_WORKERS = 4  # For data loading

    # Data Processing Limits
    # Limit dataset size for debugging/runtime constraints (set to None for full run)
    # Given 24h limit, we can use a fairly large sample or all data.
    # Setting a limit to ensure we fit within memory/time constraints during dev.
    TRAIN_SAMPLE_SIZE = 50000
    VAL_SAMPLE_SIZE = 10000

    # --------------------------------------------------------------------------
    # Text Processing Hyperparameters
    # --------------------------------------------------------------------------
    VOCAB_SIZE = 40000
    EMBEDDING_DIM = 100  # Dimension for GloVe or similar embeddings
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"

    # Sequence Lengths
    MAX_Q_LEN = 30  # Max tokens for question
    MAX_DOC_LEN = 512  # Max tokens for a candidate paragraph/context

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------

    # Ranker (ANBoW) Settings
    RANKER_HIDDEN_DIM = 128
    RANKER_DROPOUT = 0.3

    # Reader (Conv-BiDAF) Settings
    READER_CONV_KERNEL_SIZE = 5
    READER_HIDDEN_DIM = 128
    READER_DROPOUT = 0.2

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 2

    # Inference Thresholds
    # If Ranker Prob * Reader Prob < THRESHOLD, predict NULL
    INFERENCE_THRESHOLD = 0.1

    @staticmethod
    def setup_directories():
        """
        Ensures that necessary working directories exist.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        print(f"Directories ensured: {Config.WORKING_DIR}, {Config.SUBMISSION_DIR}")


# Initialize directories on import
Config.setup_directories()
