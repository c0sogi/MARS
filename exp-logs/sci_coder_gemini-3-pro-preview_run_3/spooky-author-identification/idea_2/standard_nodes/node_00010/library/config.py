import os
import torch


class Config:
    """
    Configuration class for the Authorship Attribution pipeline.
    Acts as a single source of truth for paths, hyperparameters, and global settings.
    """

    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use in debug mode
    NUM_WORKERS = 2  # Number of workers for data loading

    # ==========================================
    # Compute Environment
    # ==========================================
    # Automatically detect GPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # File Paths & Directories
    # ==========================================
    # Input Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Output Directories
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_3")
    SUBMISSION_DIR = "./submission"

    # Data Files (using metadata splits)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Model Artifacts
    MODEL_SAVE_PATH = os.path.join(CACHE_DIR, "best_transformer_model.pth")
    VECTORIZER_SAVE_PATH = os.path.join(CACHE_DIR, "tfidf_vectorizer.pkl")
    STATISTICAL_MODEL_SAVE_PATH = os.path.join(CACHE_DIR, "statistical_ensemble.pkl")

    # Submission Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure necessary directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Definitions
    # ==========================================
    # Target Mapping
    LABEL_MAP = {"EAP": 0, "HPL": 1, "MWS": 2}
    ID2LABEL = {0: "EAP", 1: "HPL", 2: "MWS"}
    NUM_CLASSES = 3
    CLASS_NAMES = ["EAP", "HPL", "MWS"]

    # ==========================================
    # Transformer Branch Hyperparameters
    # ==========================================
    # Model Architecture
    MODEL_NAME = "microsoft/deberta-v3-base"

    # Tokenization
    MAX_LEN = 256  # Sufficient for the majority of texts (mean word count ~26)

    # Training
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 32
    EPOCHS = 4
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0
    GRAD_ACCUM_STEPS = 1

    # Optimization
    EARLY_STOPPING_PATIENCE = 2  # Stop if validation loss doesn't improve for 2 epochs

    # ==========================================
    # Statistical Branch Hyperparameters
    # ==========================================
    # TF-IDF Settings
    TFIDF_PARAMS = {
        "analyzer": "word",
        "token_pattern": r"\w{1,}",
        "ngram_range": (1, 3),  # Unigrams, Bigrams, and Trigrams (Cite Lesson 2)
        "max_features": 25000,  # Increased vocabulary size for trigrams
        "sublinear_tf": True,  # Apply sublinear scaling (1 + log(tf))
        "strip_accents": "unicode",
    }

    # Character N-gram Settings (Separate vectorizer usually merged)
    CHAR_TFIDF_PARAMS = {
        "analyzer": "char",
        "ngram_range": (3, 5),  # Character 3-grams to 5-grams
        "max_features": 30000,
        "sublinear_tf": True,
    }

    # Ensemble Weights (Initial guess, can be optimized)
    # Weight for Transformer probability, (1 - weight) for Statistical
    TRANSFORMER_WEIGHT = 0.6
