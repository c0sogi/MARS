import os


class Config:
    # --------------------------------------------------------------------------
    # Global Settings & Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42

    # --------------------------------------------------------------------------
    # Directory Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for caching processed data and models (Idea 2 specific)
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # --------------------------------------------------------------------------
    # File Paths
    # --------------------------------------------------------------------------
    # Raw Input Files
    TRAIN_FILE = "simplified-nq-train.jsonl"
    TEST_FILE_PATTERN = (
        "*test.jsonl"  # Matches simplified-nq-test.jsonl or kaggle variant
    )
    SAMPLE_SUBMISSION_FILE = "sample_submission.csv"

    # Metadata Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache Files (Parquet for DataFrames, NPY for Arrays)
    VOCAB_CACHE_PATH = os.path.join(WORKING_DIR, "vocab.parquet")
    # Processed datasets for Ranker
    RANKER_TRAIN_CACHE = os.path.join(WORKING_DIR, "ranker_train_data.parquet")
    RANKER_VAL_CACHE = os.path.join(WORKING_DIR, "ranker_val_data.parquet")
    # Processed datasets for Reader
    READER_TRAIN_CACHE = os.path.join(WORKING_DIR, "reader_train_data.parquet")
    READER_VAL_CACHE = os.path.join(WORKING_DIR, "reader_val_data.parquet")

    # Model Checkpoints
    RANKER_MODEL_PATH = os.path.join(WORKING_DIR, "ranker_best.pth")
    READER_MODEL_PATH = os.path.join(WORKING_DIR, "reader_best.pth")

    # Final Submission Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Preprocessing & Vocabulary
    # --------------------------------------------------------------------------
    VOCAB_SIZE = 20000  # Maximum size of vocabulary
    EMBED_DIM = 100  # Dimension of word embeddings
    MAX_Q_LEN = 32  # Maximum sequence length for Questions
    MAX_DOC_LEN = 384  # Maximum sequence length for Paragraphs (Long Answers)

    # Special Tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    # --------------------------------------------------------------------------
    # Model Architecture: Ranker (Siamese TextCNN)
    # --------------------------------------------------------------------------
    CNN_FILTERS = 100  # Number of filters per kernel size
    CNN_KERNEL_SIZES = [
        2,
        3,
        4,
    ]  # Convolutional window sizes (Bigrams, Trigrams, 4-grams)
    RANKER_HIDDEN_DIM = 128  # Hidden dimension for the classification MLP
    RANKER_DROPOUT = 0.3  # Dropout rate for Ranker

    # --------------------------------------------------------------------------
    # Model Architecture: Reader (Attention-MLP)
    # --------------------------------------------------------------------------
    READER_HIDDEN_DIM = 128  # Hidden dimension for the extraction MLP
    READER_DROPOUT = 0.3  # Dropout rate for Reader

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 8
    PATIENCE = 2  # Early stopping patience (epochs without improvement)

    # Debugging / Development
    # Set to an integer (e.g., 5000) to limit dataset size for fast prototyping.
    # Set to None to use the full dataset.
    DEBUG_SAMPLE_SIZE = None

    # --------------------------------------------------------------------------
    # Inference & Post-Processing
    # --------------------------------------------------------------------------
    # Threshold for the combined confidence score (Ranker Prob * Reader Prob)
    # If score < THRESHOLD, prediction is empty/null.
    PREDICTION_THRESHOLD = 0.15

    # Maximum valid length (in tokens) for a predicted short answer span
    MAX_ANSWER_LEN = 30

    @staticmethod
    def setup_directories():
        """Ensures that working and submission directories exist."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
