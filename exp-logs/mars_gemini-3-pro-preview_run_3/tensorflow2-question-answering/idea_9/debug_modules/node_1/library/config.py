import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # Directory and File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_9"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Raw Data Files
    TRAIN_FILE = os.path.join(INPUT_DIR, "simplified-nq-train.jsonl")
    TEST_FILE = os.path.join(INPUT_DIR, "simplified-nq-test.jsonl")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Pre-generated)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache File Paths (Parquet/NPY)
    # These are used by data processing modules to store/load deterministic data
    VOCAB_CACHE = os.path.join(WORKING_DIR, "vocab.parquet")
    EMBEDDING_MATRIX_CACHE = os.path.join(WORKING_DIR, "embedding_matrix.npy")

    RANKER_TRAIN_CACHE = os.path.join(WORKING_DIR, "ranker_train_data.parquet")
    RANKER_VAL_CACHE = os.path.join(WORKING_DIR, "ranker_val_data.parquet")
    RANKER_TEST_FEATURES_CACHE = os.path.join(
        WORKING_DIR, "ranker_test_features.parquet"
    )

    READER_TRAIN_CACHE = os.path.join(WORKING_DIR, "reader_train_data.parquet")
    READER_VAL_CACHE = os.path.join(WORKING_DIR, "reader_val_data.parquet")

    # Model Checkpoints
    RANKER_MODEL_PATH = os.path.join(WORKING_DIR, "ranker_best.pth")
    READER_MODEL_PATH = os.path.join(WORKING_DIR, "reader_best.pth")

    # Final Submission Output
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Processing Hyperparameters
    # --------------------------------------------------------------------------
    # Text Processing
    MAX_Q_LEN = 30  # Maximum length for questions
    MAX_DOC_LEN = 256  # Maximum length for candidate paragraphs (Long Answer)
    VOCAB_SIZE = 40000  # Size of the vocabulary
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"

    # Dataset Construction
    NEG_RATIO = 1  # Number of negative samples per positive sample for Ranker
    SAMPLE_SIZE = None  # Set to an integer (e.g., 10000) for debugging/testing, None for full data

    # --------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # --------------------------------------------------------------------------
    # Shared
    EMBED_DIM = 100  # Dimension of word embeddings

    # Decomposable Attention Ranker
    RANKER_HIDDEN_SIZE = 200
    RANKER_DROPOUT = 0.2

    # Gated Convolutional Reader
    READER_FILTERS = 128
    READER_KERNEL_SIZE = 3
    READER_LAYERS = 4
    READER_DROPOUT = 0.2

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10
    EARLY_STOPPING_PATIENCE = 2

    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------------------------
    # Inference Hyperparameters
    # --------------------------------------------------------------------------
    CONFIDENCE_THRESHOLD = 0.4  # Threshold for predicting a non-null answer
    MAX_ANSWER_LEN = 30  # Maximum length for a generated short answer span
