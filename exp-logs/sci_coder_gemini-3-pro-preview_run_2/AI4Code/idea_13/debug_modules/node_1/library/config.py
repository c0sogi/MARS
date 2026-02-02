import os


class Config:
    """
    Configuration class for Idea 13: Stacked Hybrid Ranking with Supervised Metric Anchoring.
    """

    # --------------------------------------------------------------------------
    # Global Settings
    # --------------------------------------------------------------------------
    SEED = 42
    NUM_WORKERS = 4

    # Debugging / Sampling
    # Set DEBUG to True and DEBUG_SAMPLE_SIZE to a small number (e.g., 2000)
    # to run the pipeline on a subset of data for testing.
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 5000 if DEBUG else None

    # --------------------------------------------------------------------------
    # Directory & File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_13"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Inputs
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Final Submission Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Feature Engineering (Text Processing)
    # --------------------------------------------------------------------------
    # TF-IDF Vectorization
    VOCAB_SIZE = 60000
    NGRAM_RANGE = (1, 2)
    MIN_DF = 2
    TOKEN_PATTERN = r"(?u)\b\w\w+\b"  # Standard scikit-learn token pattern

    # Truncated SVD (Latent Semantic Analysis)
    SVD_COMPONENTS = 128
    SVD_ITER = 10
    SVD_RANDOM_STATE = SEED

    # --------------------------------------------------------------------------
    # Stage 1: Sparse Lexical Regressor (Ridge)
    # --------------------------------------------------------------------------
    RIDGE_ALPHA = 1.0
    RIDGE_SOLVER = "auto"

    # --------------------------------------------------------------------------
    # Stage 2: Supervised Metric Projector (Siamese Network)
    # --------------------------------------------------------------------------
    # Architecture
    METRIC_INPUT_DIM = SVD_COMPONENTS
    METRIC_HIDDEN_DIM = 256
    METRIC_EMBEDDING_DIM = 64  # Projection dimension for metric space
    METRIC_DROPOUT = 0.2

    # Training
    METRIC_EPOCHS = 10
    METRIC_BATCH_SIZE = 2048
    METRIC_LR = 1e-3
    METRIC_WEIGHT_DECAY = 1e-5
    METRIC_MARGIN = 0.5  # Margin for Contrastive Loss

    # Data Sampling for Metric Learning
    # Number of negative code cells to sample for each (markdown, adjacent_code) pair
    METRIC_NEGATIVES_PER_POSITIVE = 3

    # --------------------------------------------------------------------------
    # Stage 3: Neighborhood Gradient Booster (LightGBM)
    # --------------------------------------------------------------------------
    # Feature Engineering
    TOP_K_ANCHORS = 20  # Number of nearest code neighbors to aggregate features from

    # Model Hyperparameters
    LGBM_PARAMS = {
        "objective": "regression_l1",  # Minimize MAE
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_estimators": 3000,
        "verbose": -1,
        "random_state": SEED,
        "n_jobs": -1,
    }

    LGBM_EARLY_STOPPING_ROUNDS = 100

    # --------------------------------------------------------------------------
    # Artifact Caching (Paths for saving/loading intermediate files)
    # --------------------------------------------------------------------------
    # Saved Models
    TFIDF_PATH = os.path.join(WORKING_DIR, "tfidf_vectorizer.joblib")
    SVD_PATH = os.path.join(WORKING_DIR, "svd_model.joblib")
    RIDGE_PATH = os.path.join(WORKING_DIR, "ridge_model.joblib")
    METRIC_MODEL_PATH = os.path.join(WORKING_DIR, "metric_model.pth")

    # Processed Dataframes (Raw text extracted from JSONs)
    TRAIN_DATAFRAME_PATH = os.path.join(WORKING_DIR, "train_dataframe.parquet")
    VAL_DATAFRAME_PATH = os.path.join(WORKING_DIR, "val_dataframe.parquet")
    TEST_DATAFRAME_PATH = os.path.join(WORKING_DIR, "test_dataframe.parquet")

    # Generated Features (SVD vectors, Ridge predictions, etc.)
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # Intermediate Predictions
    TRAIN_RIDGE_OOF_PATH = os.path.join(WORKING_DIR, "train_ridge_oof.parquet")
    VAL_RIDGE_PREDS_PATH = os.path.join(WORKING_DIR, "val_ridge_preds.parquet")
    TEST_RIDGE_PREDS_PATH = os.path.join(WORKING_DIR, "test_ridge_preds.parquet")
