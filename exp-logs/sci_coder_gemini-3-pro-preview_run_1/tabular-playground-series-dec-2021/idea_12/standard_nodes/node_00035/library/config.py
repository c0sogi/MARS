import os


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Paths to the metadata-processed datasets
    # These files are used to identify the correct splits and filtered data
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output directories
    # idea_12 is the designated directory for this experiment's cache
    IDEA_DIR = os.path.join(WORKING_DIR, "idea_12")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure output directories exist
    os.makedirs(IDEA_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Global Configuration
    # ==========================================
    SEED = 42
    ID_COL = "Id"
    TARGET_COL = "Cover_Type"

    # Debugging flag to subsample data for rapid prototyping if needed
    DEBUG = False
    DEBUG_SAMPLES = 50000

    # ==========================================
    # Cross-Validation Configuration
    # ==========================================
    # Stratified 5-Fold Cross-Validation
    N_FOLDS = 5

    # ==========================================
    # Model Configuration (XGBoost)
    # ==========================================
    # Hyperparameters for the XGBoost Classifier
    XGB_PARAMS = {
        "booster": "gbtree",
        "objective": "multi:softmax",
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": "cuda",  # GPU acceleration
        "max_depth": 10,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": 12,
        "random_state": SEED,
        "verbosity": 0,
        # num_class will be determined dynamically based on label encoding
    }

    # Training Loop Parameters
    NUM_BOOST_ROUND = 5000
    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_EVAL = 100

    # ==========================================
    # Feature Engineering Configuration
    # ==========================================
    # k-NN Parameters for Manifold-Aware Feature Injection
    KNN_K = 100
    KNN_BATCH_SIZE = 2048  # Batch size for GPU distance calculation

    # Continuous features to be used for the k-NN search (Manifold Coordinate)
    KNN_FEATURES = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]
