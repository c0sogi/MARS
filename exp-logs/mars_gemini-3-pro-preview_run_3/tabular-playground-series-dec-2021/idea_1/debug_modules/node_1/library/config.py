import os


class Config:
    """
    Global configuration for the Cover Type Prediction pipeline.
    """

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    # Input Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Data Paths (using pre-generated parquet metadata)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Directories
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"

    # Output Files
    MODEL_OUTPUT_PATH = os.path.join(WORKING_DIR, "xgb_model.json")
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Dataset Information
    # -------------------------------------------------------------------------
    ID_COL = "Id"
    TARGET_COL = "Cover_Type"

    # Target classes are integers (e.g., 1-7).
    # We define the expected number of classes for the model.
    NUM_CLASSES = 7

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    # Parameters for Gradient Boosting Decision Tree (XGBoost)
    # configured for GPU acceleration and histogram-based training.
    MODEL_PARAMS = {
        "objective": "multi:softmax",
        "num_class": NUM_CLASSES,
        "tree_method": "hist",
        "device": "cuda",
        "learning_rate": 0.1,
        "max_depth": 10,
        "n_estimators": 2000,  # Max boosting rounds
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": SEED,
        "n_jobs": 12,  # Utilize available vCPUs
        "verbosity": 1,  # 0 (silent) - 3 (debug)
    }

    # -------------------------------------------------------------------------
    # Training Settings
    # -------------------------------------------------------------------------
    EARLY_STOPPING_ROUNDS = 20
    VERBOSE_EVAL = 50  # Frequency of validation metric logging

    # -------------------------------------------------------------------------
    # Debug / Development Options
    # -------------------------------------------------------------------------
    # If DEBUG is True, the pipeline will use a small subset of data
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 10000

    @classmethod
    def setup_directories(cls):
        """
        Creates the necessary working and submission directories if they do not exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
