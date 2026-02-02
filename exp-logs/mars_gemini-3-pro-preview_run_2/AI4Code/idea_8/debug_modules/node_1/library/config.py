import os
import torch


class Config:
    """
    Central configuration for the Dual-View Stacked Ranking with Lexical and Semantic Anchoring.
    Defines paths, constants, and hyperparameters.
    """

    # --------------------------------------------------------------------------
    # Directory & File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_8"
    SUBMISSION_DIR = "./submission"

    # Metadata Paths (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths for Processed Data (Parquet)
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")

    # Model Artifact Paths
    TFIDF_VECTORIZER_PATH = os.path.join(WORKING_DIR, "tfidf_vectorizer.joblib")
    LSA_MODEL_PATH = os.path.join(WORKING_DIR, "lsa_model.joblib")
    RIDGE_MODEL_PATH = os.path.join(WORKING_DIR, "ridge_model.joblib")
    LGBM_MODEL_PATH = os.path.join(WORKING_DIR, "lgbm_model.txt")

    # --------------------------------------------------------------------------
    # Global Settings
    # --------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For data loading if applicable

    # --------------------------------------------------------------------------
    # Feature Engineering Hyperparameters
    # --------------------------------------------------------------------------
    # TF-IDF Vectorizer Settings
    # High vocabulary size to capture specific code tokens and markdown terms
    TFIDF_MAX_FEATURES = 60000
    TFIDF_NGRAM_RANGE = (1, 2)
    TFIDF_SUBLINEAR_TF = True
    TFIDF_USE_IDF = True
    TFIDF_STRIP_ACCENTS = None  # "No Accent Stripping" per description

    # LSA (Latent Semantic Analysis) Settings
    LSA_COMPONENTS = 128

    # Sentence Transformer Settings (Dense Embeddings)
    # Lightweight model for efficient semantic matching
    SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
    SENTENCE_TRANSFORMER_BATCH_SIZE = 32

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------
    # Stage 1: Ridge Regression (The "Signpost" Model)
    RIDGE_ALPHA = 1.0

    # Stage 2: LightGBM (The "Refinement" Model)
    # Minimizing MAE to align with the rank-based nature of the task
    @staticmethod
    def get_lgbm_params(overrides=None):
        """
        Returns the dictionary of LightGBM parameters.
        Allows for runtime overrides for tuning.
        """
        params = {
            "objective": "mae",
            "metric": "mae",
            "boosting_type": "gbdt",
            "n_estimators": 2000,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbosity": -1,  # Silent execution
            "n_jobs": -1,
            "random_state": Config.SEED,
            # Note: early_stopping_rounds is usually passed to .train(), not params,
            # but kept here for reference if using sklearn API
        }
        if overrides:
            params.update(overrides)
        return params

    # Training Loop Settings
    NUM_FOLDS = 5
    EARLY_STOPPING_ROUNDS = 50

    # --------------------------------------------------------------------------
    # Setup Utility
    # --------------------------------------------------------------------------
    @classmethod
    def setup(cls):
        """
        Ensures that the working and submission directories exist.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
