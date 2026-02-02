import os


class Config:
    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"

    # Input Data Files
    TRAIN_DATA_PATH = os.path.join(INPUT_DIR, "simplified-nq-train.jsonl")
    TEST_DATA_PATH = os.path.join(INPUT_DIR, "simplified-nq-test.jsonl")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache Files (For deterministic processing)
    # We use .npy for numpy arrays and .parquet for dataframes to avoid pickle
    VOCAB_CACHE_PATH = os.path.join(
        WORKING_DIR, "vocab.json"
    )  # Save vocab mapping as JSON
    IDF_CACHE_PATH = os.path.join(WORKING_DIR, "idf_weights.npy")  # Save TF-IDF weights
    EMBEDDING_MATRIX_PATH = os.path.join(WORKING_DIR, "embedding_matrix.npy")

    # Model Checkpoint
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "boe_ranker_model.pt")
    SUBMISSION_SAVE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # Data Processing & Tokenization
    # ==========================================
    # Text Processing
    MAX_SEQ_LEN = 128  # Maximum tokens for Question/Candidate encoding
    VOCAB_SIZE = 40000  # Max vocabulary size
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"

    # Short Answer Heuristic
    SHORT_ANSWER_WINDOW_SIZE = 25  # Number of tokens in sliding window
    SHORT_ANSWER_STRIDE = 5  # Stride for sliding window

    # ==========================================
    # Model Architecture (Neural BoE Ranker)
    # ==========================================
    EMBEDDING_DIM = 100  # Dimension of word embeddings
    HIDDEN_DIMS = [256, 128]  # Hidden layers for the MLP classifier
    DROPOUT_RATE = 0.3  # Dropout probability
    FREEZE_EMBEDDINGS = False  # Whether to freeze embedding layer during training

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5
    PATIENCE = 2  # Early stopping patience
    NEGATIVE_SAMPLES_RATIO = 4  # Number of negative candidates per positive sample

    # ==========================================
    # Inference Thresholds
    # ==========================================
    TAU_LONG = 0.5  # Threshold score for predicting a long answer (vs BLANK)
    TAU_SHORT = (
        0.15  # Threshold cosine similarity for predicting a short answer (vs BLANK)
    )

    # ==========================================
    # Debugging / Development
    # ==========================================
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 2000  # Number of samples to use if DEBUG is True

    @staticmethod
    def setup():
        """Creates necessary working directories."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
