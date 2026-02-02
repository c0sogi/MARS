import os


class Config:
    # --------------------------------------------------------------------------
    # 1. Paths & Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_15"
    SUBMISSION_DIR = "./submission"

    # Metadata file paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # --------------------------------------------------------------------------
    # 2. Global Settings
    # --------------------------------------------------------------------------
    RANDOM_STATE = 42
    NUM_WORKERS = 4  # Adjust based on available vCPUs

    # --------------------------------------------------------------------------
    # 3. Preprocessing Hyperparameters
    # --------------------------------------------------------------------------
    # TF-IDF Vectorizer (Lexical View)
    VOCAB_SIZE = 60000
    NGRAM_RANGE = (1, 2)
    MIN_DF = 2
    USE_IDF = True
    SUBLINEAR_TF = True
    STRIP_ACCENTS = None  # "No Accent Stripping" per requirements

    # Truncated SVD (Latent View)
    SVD_COMPONENTS = 128
    SVD_N_ITER = 5

    # Symbolic Extraction (Data Flow View)
    # Regex to capture identifiers (variables, functions)
    # Starts with letter/underscore, followed by alphanum/underscore
    SYMBOLIC_TOKEN_PATTERN = r"[a-zA-Z_][a-zA-Z0-9_]*"

    # --------------------------------------------------------------------------
    # 4. Feature Engineering (Anchors)
    # --------------------------------------------------------------------------
    # Number of nearest neighbors to consider for anchor statistics
    TOP_K = 5

    # --------------------------------------------------------------------------
    # 5. Model Hyperparameters
    # --------------------------------------------------------------------------
    # Stage 1: Sparse Lexical Regressor
    RIDGE_ALPHA = 1.0
    RIDGE_SOLVER = "auto"

    # Stage 2: Multi-View Gradient Booster
    LGBM_PARAMS = {
        "n_estimators": 5000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "mae",  # Minimize Mean Absolute Error for Rank
        "metric": "mae",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "verbosity": -1,
        "importance_type": "gain",
    }

    # Training Loop Settings
    EARLY_STOPPING_ROUNDS = 100
    VERBOSE_EVAL = 100

    @staticmethod
    def setup():
        """
        Ensures that the necessary working and submission directories exist.
        Should be called at the start of the pipeline.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
