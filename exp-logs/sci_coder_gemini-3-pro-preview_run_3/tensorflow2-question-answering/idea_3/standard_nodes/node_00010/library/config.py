import os


class Config:
    # --------------------------------------------------------------------------
    # Global Settings
    # --------------------------------------------------------------------------
    SEED = 42
    NUM_WORKERS = 4  # Number of subprocesses for data loading

    # --------------------------------------------------------------------------
    # File Paths & Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Raw Input Files
    TRAIN_FILE = os.path.join(INPUT_DIR, "simplified-nq-train.jsonl")
    TEST_FILE = os.path.join(INPUT_DIR, "simplified-nq-test.jsonl")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Pre-generated)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache Paths (Parquet files for deterministic data processing)
    # These are used to store processed tensors/indices to speed up subsequent runs
    VOCAB_CACHE = os.path.join(WORKING_DIR, "vocab.parquet")

    # Ranker specific caches (Question + Paragraph pairs)
    RANKER_TRAIN_CACHE = os.path.join(WORKING_DIR, "ranker_train_data.parquet")
    RANKER_VAL_CACHE = os.path.join(WORKING_DIR, "ranker_val_data.parquet")

    # Reader specific caches (Question + Paragraph + Span indices)
    READER_TRAIN_CACHE = os.path.join(WORKING_DIR, "reader_train_data.parquet")
    READER_VAL_CACHE = os.path.join(WORKING_DIR, "reader_val_data.parquet")

    # Model Checkpoints
    RANKER_MODEL_PATH = os.path.join(WORKING_DIR, "ranker_best.pth")
    READER_MODEL_PATH = os.path.join(WORKING_DIR, "reader_best.pth")

    # Final Output
    SUBMISSION_OUTPUT = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Processing Hyperparameters
    # --------------------------------------------------------------------------
    # Vocabulary settings
    VOCAB_SIZE = 30000  # Maximum size of vocabulary
    MIN_FREQ = 2  # Minimum frequency for a token to be included
    UNK_TOKEN = "<UNK>"  # Token for unknown words
    PAD_TOKEN = "<PAD>"  # Token for padding

    # Sequence Lengths (Truncation/Padding limits)
    MAX_Q_LEN = 30  # Maximum tokens for the question
    MAX_CTX_LEN = 400  # Maximum tokens for a candidate paragraph (Long Answer)

    # HTML Parsing
    # Tags that define top-level candidate paragraphs in the document text
    CANDIDATE_TAGS = [
        "<P>",
        "<Table>",
        "<Ul>",
        "<Ol>",
        "<Dl>",
        "<H1>",
        "<H2>",
        "<H3>",
        "<H4>",
        "<H5>",
        "<H6>",
    ]

    # Dataset Sampling (for debugging or resource constraints)
    # Set to None to use the full dataset found in metadata
    # Set to an integer (e.g., 10000) to sample a subset
    TRAIN_SAMPLE_SIZE = 50000
    VAL_SAMPLE_SIZE = 5000

    # --------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # --------------------------------------------------------------------------
    # Embedding
    EMBEDDING_DIM = 100  # Dimension of word embeddings

    # LSTM Encoder settings
    HIDDEN_DIM = 128  # Hidden dimension size for Bi-LSTMs
    LSTM_LAYERS = 1  # Number of stacked LSTM layers
    DROPOUT = 0.3  # Dropout rate applied after embeddings and LSTM

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 32  # Batch size for training and validation
    LEARNING_RATE = 1e-3  # Initial learning rate for Adam optimizer
    EPOCHS = 5  # Maximum number of training epochs
    PATIENCE = 2  # Early stopping patience (epochs without improvement)
    GRAD_CLIP = 1.0  # Max norm for gradient clipping to prevent explosion

    # --------------------------------------------------------------------------
    # Inference Hyperparameters
    # --------------------------------------------------------------------------
    # Ranker threshold: if cosine similarity < threshold, prediction is NULL (no answer)
    # Value range: -1.0 to 1.0
    RANKER_THRESHOLD = 0.1
