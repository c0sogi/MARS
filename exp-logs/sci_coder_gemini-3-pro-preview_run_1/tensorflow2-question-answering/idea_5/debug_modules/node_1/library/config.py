import os
import torch


class Config:
    """
    Configuration for the Question-Conditioned Bi-Directional GRU (QC-BiGRU) model pipeline.
    """

    # ==========================================
    # System & Reproducibility
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2  # Number of subprocesses for data loading

    # ==========================================
    # Directories & File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"
    SUBMISSION_DIR = "./submission"

    # Raw Data
    TRAIN_DATA_PATH = os.path.join(INPUT_DIR, "simplified-nq-train.jsonl")
    TEST_DATA_PATH = os.path.join(INPUT_DIR, "simplified-nq-test.jsonl")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Splits)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Cache / Artifacts (Saved in Working Dir)
    VOCAB_PATH = os.path.join(WORKING_DIR, "vocab.parquet")
    EMBEDDING_MATRIX_PATH = os.path.join(WORKING_DIR, "embedding_matrix.npy")

    # Processed Data Cache
    PROCESSED_TRAIN_PATH = os.path.join(WORKING_DIR, "processed_train.parquet")
    PROCESSED_VAL_PATH = os.path.join(WORKING_DIR, "processed_val.parquet")
    PROCESSED_TEST_PATH = os.path.join(WORKING_DIR, "processed_test.parquet")

    # Model Checkpoints
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "qc_bigru_model.pth")

    # Output
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Preprocessing Hyperparameters
    # ==========================================
    MAX_VOCAB_SIZE = 20_000  # Limit vocabulary to top N frequent words
    EMBEDDING_DIM = 100  # Dimension of word embeddings (e.g., GloVe)
    MAX_SEQ_LEN = 256  # Truncate candidate text to this length
    MAX_Q_LEN = 32  # Truncate question text to this length
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"

    # ==========================================
    # Model Architecture Hyperparameters
    # ==========================================
    HIDDEN_SIZE = 128  # Hidden size for GRU units
    NUM_LAYERS = 1  # Number of stacked GRU layers
    DROPOUT = 0.2  # Dropout probability
    BIDIRECTIONAL = True  # Use Bi-Directional GRU

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 5
    NEGATIVE_SAMPLING_RATIO = 0.2  # Ratio of negative samples to keep in training
    EARLY_STOPPING_PATIENCE = (
        2  # Epochs to wait before stopping if val loss doesn't improve
    )
    WEIGHT_DECAY = 1e-5  # L2 Regularization

    # ==========================================
    # Inference / Evaluation Hyperparameters
    # ==========================================
    # Threshold for predicting a Long Answer (vs NULL)
    LONG_ANSWER_THRESHOLD = 0.4

    # Threshold for predicting a Short Answer span (vs Long Answer only)
    # Sum of start_logit + end_logit
    SHORT_ANSWER_THRESHOLD = 1.0

    # ==========================================
    # Debugging / Development
    # ==========================================
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLE_SIZE = 1000  # Number of samples to use if DEBUG is True

    @classmethod
    def setup(cls):
        """
        Creates necessary directories for artifacts and submissions.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        if cls.DEBUG:
            print(
                f"[Config] Debug mode ENABLED. Using {cls.DEBUG_SAMPLE_SIZE} samples."
            )

        print(f"[Config] Device: {cls.DEVICE}")
        print(f"[Config] Working Directory: {cls.WORKING_DIR}")


# Initialize directories on import
Config.setup()
