import os


class Config:
    """
    Configuration for the Stacked Hybrid Ranking with Multi-View Instance-Based Anchoring.
    Centralizes all hyperparameters, file paths, and global settings.
    """

    # --------------------------------------------------------------------------
    # Global Configuration
    # --------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging
    DEBUG_SAMPLE_SIZE = 2000  # Number of notebooks to use in debug mode
    NUM_WORKERS = 4  # Number of CPU workers for parallel data processing

    # --------------------------------------------------------------------------
    # Directory Structure
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_19"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # --------------------------------------------------------------------------
    # Preprocessing & Vectorization (Stage 1)
    # --------------------------------------------------------------------------
    # TF-IDF Configuration
    # Based on lessons: 60k vocab, bigrams, sublinear TF, no accent stripping
    TFIDF_PARAMS = {
        "input": "content",
        "encoding": "utf-8",
        "decode_error": "strict",
        "strip_accents": None,  # Explicitly None per instructions
        "lowercase": True,
        "preprocessor": None,
        "tokenizer": None,
        "analyzer": "word",
        "stop_words": "english",  # Standard practice for code/markdown mix
        "token_pattern": r"(?u)\b\w\w+\b",
        "ngram_range": (1, 2),
        "max_df": 0.9,
        "min_df": 2,
        "max_features": 60000,
        "norm": "l2",
        "use_idf": True,
        "smooth_idf": True,
        "sublinear_tf": True,
    }

    # Latent Semantic Analysis (SVD) Configuration
    # Based on lessons: 128 components for dense representation
    SVD_COMPONENTS = 128
    SVD_RANDOM_STATE = SEED

    # --------------------------------------------------------------------------
    # Feature Engineering (Stage 2)
    # --------------------------------------------------------------------------
    # Multi-View Instance Features
    # View 1: Lexical (Sparse TF-IDF Cosine)
    # View 2: Latent (Dense SVD Cosine)
    # View 3: Symbolic (Jaccard of Identifiers)

    # Number of neighbors to use for smoothed aggregate statistics (Mean/Std)
    K_NEIGHBORS_SMOOTH = 5

    # Number of top neighbors to extract as explicit individual features
    # (e.g., Rank of 1st neighbor, Similarity of 1st neighbor, etc.)
    K_NEIGHBORS_EXPLICIT = 3

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------
    # Stage 1: Ridge Regression (The "Signpost" Model)
    RIDGE_ALPHA = 1.0
    RIDGE_RANDOM_STATE = SEED

    # Stage 2: LightGBM (The "Refinement" Model)
    # Objective: Minimize Mean Absolute Error (MAE)
    LGBM_PARAMS = {
        "objective": "regression_l1",  # MAE objective
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "n_jobs": -1,
        "verbosity": -1,
        "seed": SEED,
        "force_col_wise": True,
    }

    # Training Loop Parameters
    NUM_BOOST_ROUND = 5000
    EARLY_STOPPING_ROUNDS = 100
    VERBOSE_EVAL = 100

    # --------------------------------------------------------------------------
    # Caching & Artifacts
    # --------------------------------------------------------------------------
    # Saved Models
    PATH_TFIDF_VECTORIZER = os.path.join(WORKING_DIR, "tfidf_vectorizer.joblib")
    PATH_SVD_MODEL = os.path.join(WORKING_DIR, "svd_model.joblib")
    PATH_RIDGE_MODEL = os.path.join(WORKING_DIR, "ridge_model.joblib")
    PATH_LGBM_MODEL = os.path.join(WORKING_DIR, "lgbm_model.txt")

    # Cached Dataframes (Parquet)
    # These store the processed text, ranks, and metadata
    CACHE_TRAIN_DATAFRAME = os.path.join(WORKING_DIR, "train_dataframe.parquet")
    CACHE_VAL_DATAFRAME = os.path.join(WORKING_DIR, "val_dataframe.parquet")
    CACHE_TEST_DATAFRAME = os.path.join(WORKING_DIR, "test_dataframe.parquet")

    # Cached Features (Parquet)
    # These store the heavy Multi-View Instance features
    CACHE_TRAIN_FEATURES = os.path.join(WORKING_DIR, "train_features.parquet")
    CACHE_VAL_FEATURES = os.path.join(WORKING_DIR, "val_features.parquet")
    CACHE_TEST_FEATURES = os.path.join(WORKING_DIR, "test_features.parquet")

    # Intermediate Predictions
    CACHE_STAGE1_OOF_PREDS = os.path.join(WORKING_DIR, "stage1_oof_preds.parquet")
    CACHE_TEST_RIDGE_PREDS = os.path.join(WORKING_DIR, "test_ridge_preds.parquet")

    # Final Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
