import os
import torch


class Config:
    """
    Global configuration for the Question Answering project.
    """

    # --------------------------------------------------------------------------
    # Directory & File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4/"
    SUBMISSION_DIR = "./submission"

    # Source Data
    TRAIN_FILE = os.path.join(INPUT_DIR, "simplified-nq-train.jsonl")
    TEST_FILE = os.path.join(INPUT_DIR, "simplified-nq-test.jsonl")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache Files (Parquet/NPY)
    VOCAB_CACHE_PATH = os.path.join(WORKING_DIR, "vocab.parquet")
    RANKER_TRAIN_CACHE = os.path.join(WORKING_DIR, "ranker_train_data.parquet")
    RANKER_VAL_CACHE = os.path.join(WORKING_DIR, "ranker_val_data.parquet")
    READER_TRAIN_CACHE = os.path.join(WORKING_DIR, "reader_train_data.parquet")
    READER_VAL_CACHE = os.path.join(WORKING_DIR, "reader_val_data.parquet")

    # Model Checkpoints
    RANKER_MODEL_PATH = os.path.join(WORKING_DIR, "ranker_best.pth")
    READER_MODEL_PATH = os.path.join(WORKING_DIR, "reader_best.pth")

    # Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Reproducibility & Hardware
    # --------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # --------------------------------------------------------------------------
    # Data Processing Hyperparameters
    # --------------------------------------------------------------------------
    # Vocabulary
    MAX_VOCAB_SIZE = 30000
    MIN_FREQ = 2
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"

    # Sequence Lengths
    MAX_Q_LEN = 30  # Max tokens for question
    MAX_DOC_LEN = 256  # Max tokens for a candidate paragraph (Ranker)
    MAX_READER_SEQ_LEN = 300  # Combined Q + Candidate length (Reader)

    # Sampling (for debugging/speed)
    # Set to None to use full dataset, or an integer to limit samples
    DEBUG_SAMPLE_SIZE = None

    # Negative Sampling for Ranker
    NUM_NEGATIVES_PER_POSITIVE = 3

    # --------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # --------------------------------------------------------------------------
    EMBEDDING_DIM = 128  # Dimension for word embeddings

    # Ranker (Siamese Self-Attention)
    RANKER_HEADS = 4
    RANKER_LAYERS = 1
    RANKER_FF_DIM = 256
    RANKER_DROPOUT = 0.1

    # Reader (Separable ConvNet)
    READER_FILTERS = 128
    READER_KERNEL_SIZE = 7  # Large kernel for n-gram context
    READER_LAYERS = 4
    READER_DROPOUT = 0.2

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 64  # A100 can handle larger batches
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # AdamW weight decay
    EPOCHS = 10
    PATIENCE = 3  # Early stopping patience

    # --------------------------------------------------------------------------
    # Inference Hyperparameters
    # --------------------------------------------------------------------------
    CONFIDENCE_THRESHOLD = 0.1  # Threshold to predict an answer vs NULL
    TOP_K_CANDIDATES = 1  # Number of candidates to pass to Reader

    def __init__(self):
        """
        Initialize configuration and ensure necessary directories exist.
        """
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)

        # Set seeds for reproducibility
        torch.manual_seed(self.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.SEED)
