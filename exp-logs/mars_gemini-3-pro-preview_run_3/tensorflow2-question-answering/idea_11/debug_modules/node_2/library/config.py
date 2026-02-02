import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # Directory & File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_11"
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Raw Data
    TRAIN_DATA_FILE = os.path.join(INPUT_DIR, "simplified-nq-train.jsonl")
    TEST_DATA_FILE = os.path.join(INPUT_DIR, "simplified-nq-test.jsonl")
    SAMPLE_SUBMISSION_FILE = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cached Artifacts (Parquet/NPY/PTH)
    VOCAB_PATH = os.path.join(WORKING_DIR, "vocab.parquet")
    EMBEDDING_MATRIX_PATH = os.path.join(WORKING_DIR, "embedding_matrix.npy")

    # Processed Datasets
    RANKER_TRAIN_DATA = os.path.join(WORKING_DIR, "ranker_train_data.parquet")
    RANKER_VAL_DATA = os.path.join(WORKING_DIR, "ranker_val_data.parquet")
    READER_TRAIN_DATA = os.path.join(WORKING_DIR, "reader_train_data.parquet")
    READER_VAL_DATA = os.path.join(WORKING_DIR, "reader_val_data.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # Model Checkpoints
    RANKER_MODEL_PATH = os.path.join(WORKING_DIR, "ranker_best.pth")
    READER_MODEL_PATH = os.path.join(WORKING_DIR, "reader_best.pth")
    FINAL_SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # General Configuration
    # --------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Debugging / Development
    # Set to True to run on a small subset of data for quick verification
    DEBUG = False
    # Number of samples to use if DEBUG is True (or None for full dataset)
    TRAIN_SUBSET_SIZE = 5000 if DEBUG else None
    VAL_SUBSET_SIZE = 1000 if DEBUG else None

    # --------------------------------------------------------------------------
    # Data Preprocessing Hyperparameters
    # --------------------------------------------------------------------------
    VOCAB_SIZE = 40000  # Max vocabulary size
    EMBEDDING_DIM = 100  # Dimension of word embeddings (e.g., GloVe)
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"

    # Sequence Lengths
    MAX_Q_LEN = 30  # Max query length
    MAX_CTX_LEN = 300  # Max length for a candidate paragraph

    # Candidate Generation
    # HTML tags used to split document into paragraphs
    CANDIDATE_TAGS = ["<P>", "<Table>", "<Ul>", "<Ol>", "<H1>", "<H2>", "<H3>"]

    # Ranker Training Data Construction
    NUM_NEGATIVES = 1  # Number of hard negatives per positive for ranking

    # --------------------------------------------------------------------------
    # Model Architectures
    # --------------------------------------------------------------------------

    # Histogram-Based Matching Ranker
    HISTOGRAM_BINS = 11  # Number of bins for cosine similarity [-1, 1]
    RANKER_HIDDEN_DIM = 64  # Hidden size for the scoring MLP
    RANKER_DROPOUT = 0.1

    # Quasi-Recurrent (QRNN) Reader
    QRNN_KERNEL_SIZE = 3  # Convolution kernel size
    QRNN_HIDDEN_DIM = 128  # Hidden dimension size
    QRNN_NUM_LAYERS = 2  # Number of QRNN layers
    QRNN_DROPOUT = 0.2  # Dropout probability within QRNN

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 10
    EARLY_STOPPING_PATIENCE = 3  # Stop after N epochs of no validation improvement

    # Loss Configuration
    RANKER_MARGIN = 0.5  # Margin for Pairwise Hinge Loss

    # --------------------------------------------------------------------------
    # Inference Hyperparameters
    # --------------------------------------------------------------------------
    MAX_TEST_CANDIDATES = 15  # Max paragraphs to process per document during inference
    CONFIDENCE_THRESHOLD = 0.4  # Threshold for predicting a short answer vs NULL
    MAX_ANSWER_LEN = 30  # Maximum length of a predicted short answer span
