import os


class Config:
    """
    Configuration for the NBSVM (Naive Bayes - Support Vector Machine) model
    with Bias-Centric Sample Weighting.
    """

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42

    # ==========================================
    # File Paths
    # ==========================================
    # Metadata Inputs (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Working Directory for Caching Intermediate Artifacts (Idea 2)
    WORKING_DIR = "./working/idea_2"

    # Cached Feature Paths
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.npz")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.npz")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.npz")

    # Cached Weights and Model
    TRAIN_WEIGHTS_PATH = os.path.join(WORKING_DIR, "train_weights.npy")
    MODEL_PATH = os.path.join(WORKING_DIR, "nbsvm_model.pkl")

    # ==========================================
    # Column Definitions
    # ==========================================
    TEXT_COL = "comment_text"
    TARGET_COL = "target"
    BINARY_TARGET_COL = "binary_target"

    # Identity columns used for the competition metric and bias weighting
    IDENTITY_COLUMNS = [
        "male",
        "female",
        "homosexual_gay_or_lesbian",
        "christian",
        "jewish",
        "muslim",
        "black",
        "white",
        "psychiatric_or_mental_illness",
    ]

    # ==========================================
    # Model Hyperparameters (NBSVM)
    # ==========================================
    # TF-IDF Vectorizer Settings
    # NBSVM typically performs well with a mix of word and character n-grams.

    # Word n-grams (Unigrams + Bigrams)
    WORD_NGRAM_RANGE = (1, 2)
    WORD_MAX_FEATURES = None  # Use all features (sparse matrix handles memory)
    WORD_MIN_DF = 3  # Prune very rare words

    # Character n-grams (2 to 6 chars)
    CHAR_NGRAM_RANGE = (2, 6)
    CHAR_MAX_FEATURES = 50000  # Limit char features to control dimensionality
    CHAR_MIN_DF = 3

    # Logistic Regression Settings (The SVM part of NBSVM)
    C = 1.0  # Inverse of regularization strength
    SOLVER = "lbfgs"  # Solver suitable for large datasets
    MAX_ITER = 1000  # Maximum iterations for convergence
    N_JOBS = -1  # Use all available cores

    # ==========================================
    # Bias-Centric Sample Weighting Hyperparameters
    # ==========================================
    # We assign higher weights to examples that are historically difficult
    # for models due to bias (BPSN and BNSP scenarios).

    # Weight multiplier for:
    # 1. Background Positive, Subgroup Negative (Non-toxic + Identity Mention)
    # 2. Background Negative, Subgroup Positive (Toxic + Identity Mention)
    BIAS_WEIGHT_MULTIPLIER = 5.0

    # Base weight for standard examples
    BASE_WEIGHT = 1.0

    @classmethod
    def setup(cls):
        """
        Creates the necessary directories for submission and working files.
        """
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
