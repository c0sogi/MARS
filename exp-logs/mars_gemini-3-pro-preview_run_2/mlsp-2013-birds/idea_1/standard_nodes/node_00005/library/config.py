import os


class Config:
    """
    Central configuration for the Bird Species Classification project.
    Contains file paths, directory structures, and model hyperparameters.
    """

    # ==========================================
    # Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    SUPPLEMENTAL_DIR = os.path.join(INPUT_DIR, "supplemental_data")

    # Output directories
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"

    # Create output directories if they don't exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # File Paths
    # ==========================================
    # Metadata files (generated previously)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Raw Feature Data
    HISTOGRAM_FILE = os.path.join(SUPPLEMENTAL_DIR, "histogram_of_segments.txt")

    # Submission Files
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache File for processed features
    CACHE_FILE = os.path.join(WORKING_DIR, "processed_features.parquet")

    # ==========================================
    # Global Constants
    # ==========================================
    NUM_SPECIES = 19
    RANDOM_SEED = 42

    # ==========================================
    # Model Hyperparameters (Random Forest)
    # ==========================================
    # These parameters are chosen to balance bias and variance for a small dataset
    # while handling the multi-label nature via class weighting.
    RF_PARAMS = {
        "n_estimators": 500,  # Number of trees in the forest
        "max_depth": 15,  # Limit depth to prevent memorization
        "min_samples_split": 5,  # Minimum samples required to split an internal node
        "min_samples_leaf": 2,  # Minimum samples required to be at a leaf node
        "class_weight": "balanced",  # Automatically adjust weights inversely proportional to class frequencies
        "random_state": RANDOM_SEED,  # Reproducibility
        "n_jobs": -1,  # Use all available cores
        "verbose": 0,  # Silent mode
    }
