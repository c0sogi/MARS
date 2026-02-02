import os


class Config:
    """
    Configuration class for the Author Identification pipeline.
    Defines paths, hyperparameters, and constants used across the project.
    """

    # ==========================================
    # General Setup
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to train on a small subset for debugging
    DEBUG_SAMPLES = 2000  # Number of samples to use in debug mode

    # ==========================================
    # File Paths
    # ==========================================
    # Input Data (Metadata)
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Dataset Schema
    # ==========================================
    ID_COL = "id"
    TEXT_COL = "text"
    TARGET_COL = "author"
    CLASSES = ["EAP", "HPL", "MWS"]

    # ==========================================
    # Feature Extraction (TF-IDF)
    # ==========================================
    # Word-level TF-IDF hyperparameters
    # Captures vocabulary, common phrases, and semantic content.
    WORD_TFIDF_PARAMS = {
        "ngram_range": (1, 3),  # Unigrams, Bigrams, and Trigrams
        "analyzer": "word",
        "token_pattern": r"\w{1,}",  # Tokenize alphanumeric strings
        "stop_words": None,  # Keep stop words as they are stylistically significant
        "sublinear_tf": True,  # Apply logarithmic scaling to term frequency (1 + log(tf))
        "lowercase": False,  # Preserve capitalization to capture stylistic choices
        "strip_accents": None,  # Preserve accents
        "min_df": 2,  # Ignore terms appearing in fewer than 2 documents
        "max_features": None,  # Use full vocabulary (high dimensionality is handled by LR)
    }

    # Character-level TF-IDF hyperparameters
    # Captures sub-word structures, punctuation habits, and morphology.
    CHAR_TFIDF_PARAMS = {
        "ngram_range": (2, 5),  # Character n-grams from length 2 to 5
        "analyzer": "char",
        "sublinear_tf": True,
        "lowercase": False,  # Preserve case
        "min_df": 2,
        "max_features": None,
    }

    # ==========================================
    # Model Hyperparameters (Logistic Regression)
    # ==========================================
    MODEL_PARAMS = {
        "C": 1.0,  # Inverse of regularization strength (Standard L2)
        "solver": "saga",  # Efficient solver for large, sparse datasets
        "multi_class": "multinomial",  # Minimizes Multi-class Logarithmic Loss
        "penalty": "l2",  # Ridge regularization
        "n_jobs": -1,  # Use all available vCPUs
        "random_state": SEED,
        "max_iter": 1000,  # Max iterations for convergence
        "tol": 1e-4,  # Tolerance for stopping criteria
    }
