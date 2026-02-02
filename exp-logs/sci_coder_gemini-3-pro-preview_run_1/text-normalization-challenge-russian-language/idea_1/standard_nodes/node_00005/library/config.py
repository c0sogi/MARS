import os


class Config:
    """
    Central configuration for the Text Normalization project.
    Contains file paths, data schema definitions, special tokens, and hyperparameters.
    """

    # --- Reproducibility ---
    SEED = 42

    # --- Debugging ---
    # If set to an integer, only this many samples will be used for training/validation
    # Set to None for full dataset training
    MAX_TRAIN_SAMPLES = None

    # --- Paths ---
    # Input Metadata Directories (Pre-split data)
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Raw Input Directory (for reference or sample submission)
    INPUT_DIR = "./input"
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "ru_sample_submission_2.csv")

    # Working Directory (for caching intermediate processed data and model stats)
    WORKING_DIR = "./working/idea_1"

    # Submission Directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Cache File Paths ---
    # Used to store processed sequence data (parquet) and model statistics (npy)
    # This enables the deterministic caching logic required by the task.
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_sequences.parquet")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_sequences.parquet")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_sequences.parquet")
    # Updated filename to invalidate cache and force training of the new hybrid model
    MODEL_STATS_PATH = os.path.join(WORKING_DIR, "ngram_neural_stats_v2.npy")

    # --- Data Schema ---
    # Column names matching the provided datasets
    SENTENCE_ID_COL = "sentence_id"
    TOKEN_ID_COL = "token_id"
    CLASS_COL = "class"
    INPUT_COL = "before"
    TARGET_COL = "after"
    SUBMISSION_ID_COL = "id"

    # --- Special Tokens ---
    # Used for padding sentence boundaries for N-gram context
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    # --- Model Hyperparameters ---
    # Context window size for the N-gram model (1 means look at 1 prev and 1 next)
    CONTEXT_WINDOW = 1

    @classmethod
    def setup(cls):
        """
        Ensures that the working and submission directories exist.
        Should be called at the start of execution.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically setup directories when config is imported
Config.setup()
