import os


class Config:
    """
    Configuration class for Text Normalization Task (Idea 2: XGBoost + Dictionary).
    """

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42

    # ==========================================
    # Directory Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # File Paths
    # ==========================================
    # Input Data (Generated Metadata)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Sample Submission
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "en_sample_submission_2.csv")

    # Cached Processed Data (Parquet for DataFrames)
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # Model Artifacts
    MODEL_FILE = os.path.join(WORKING_DIR, "xgb_model.json")
    LABEL_ENCODER_PATH = os.path.join(WORKING_DIR, "classes.npy")

    # Normalization Dictionary (JSON for portability/readability)
    NORM_DICT_PATH = os.path.join(WORKING_DIR, "normalization_dict.json")

    # Final Submission
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Feature Engineering Hyperparameters
    # ==========================================
    # Context: Number of tokens to look at to the left and right
    CONTEXT_WINDOW = 2

    # Character N-grams for the target token
    CHAR_NGRAM_RANGE = (1, 3)

    # Max features for vectorization (to keep memory usage check)
    MAX_TEXT_FEATURES = 1000

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    # Downsample the 'PLAIN' class in training data to handle class imbalance.
    # 0.05 means we keep roughly 5% of PLAIN tokens.
    PLAIN_DOWNSAMPLE_RATIO = 0.05

    # Debugging: Set to a small integer (e.g., 10000) to limit rows during development
    # Set to None for full training
    DEBUG_ROW_LIMIT = None

    # ==========================================
    # XGBoost Hyperparameters
    # ==========================================
    # These parameters are optimized for XGBoost 3.0.5 on GPU
    XGB_PARAMS = {
        "objective": "multi:softmax",
        "tree_method": "hist",  # Efficient histogram-based algorithm
        "device": "cuda",  # Use NVIDIA A100 GPU
        "learning_rate": 0.1,
        "max_depth": 10,  # Deeper trees for complex text patterns
        "n_estimators": 1000,  # Max boosting rounds (controlled by early stopping)
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0.1,
        "random_state": SEED,
        "n_jobs": 12,  # Use available vCPUs
        "verbosity": 0,  # Silent
        "early_stopping_rounds": 50,
    }
