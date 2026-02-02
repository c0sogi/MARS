import os


class Config:
    """
    Central configuration for the Leaf Classification task.
    """

    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Input Files (using metadata as the source of truth for splits)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Directories
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"

    # Output Files
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    ID_COL = "id"
    TARGET_COL = "species"

    # Columns to exclude from the feature set during training
    # 'file_path' and 'full_path' are metadata columns, not features
    DROP_COLS = ["id", "species", "file_path", "full_path"]

    # Post-processing clipping value to avoid log loss extremes
    PROB_CLIP_EPSILON = 1e-15

    # ==========================================
    # Experiment Configuration
    # ==========================================
    RANDOM_SEED = 42
    N_FOLDS = 5

    # Debugging controls
    DEBUG = False
    DEBUG_SAMPLES = 100  # Number of samples to use if DEBUG is True

    # ==========================================
    # Model Configuration (LightGBM)
    # ==========================================
    # Hyperparameters for the LightGBM model
    LGBM_PARAMS = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "seed": RANDOM_SEED,
        "n_jobs": -1,
        # 'num_class' will be set dynamically during training
    }

    # Training loop parameters
    NUM_BOOST_ROUND = 1000
    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_EVAL = 100

    @classmethod
    def setup(cls):
        """
        Creates the necessary output directories for the experiment.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
