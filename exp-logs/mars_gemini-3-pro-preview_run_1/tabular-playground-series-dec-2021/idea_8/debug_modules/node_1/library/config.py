import os


class Config:
    """
    Global configuration for the Semi-Supervised Homogeneous Ensemble pipeline.
    Defines paths, hyperparameters, and strategy-specific settings.
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_8"
    SUBMISSION_DIR = "./submission"

    # Data Paths (using pre-split metadata)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Sample submission for format reference
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Final output path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Reproducibility
    # =========================================================================
    SEED = 42

    # =========================================================================
    # Data Configuration
    # =========================================================================
    TARGET_COL = "Cover_Type"
    ID_COL = "Id"

    # Class Mapping: XGBoost requires labels in [0, num_class-1].
    # The dataset contains classes: 1, 2, 3, 4, 6, 7 (Class 5 is missing).
    # We map these to 0-5 for training and map back for submission.
    CLASS_MAPPING = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4, 7: 5}
    INVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}
    NUM_CLASSES = len(CLASS_MAPPING)

    # =========================================================================
    # Feature Engineering Strategy
    # =========================================================================
    # Dual-Representation: We retain OHE binaries but also generate dense
    # integer indices for these column groups to capture ordinal/grouped signals.
    DENSE_PREFIXES = ["Soil_Type", "Wilderness_Area"]

    # =========================================================================
    # Model Configuration (XGBoost)
    # =========================================================================
    # Configured for GPU acceleration (A100) and high model capacity.
    XGB_PARAMS = {
        "n_estimators": 3000,  # High ceiling, controlled by early stopping
        "learning_rate": 0.05,  # eta
        "max_depth": 10,  # High capacity backbone
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "multi:softmax",
        "eval_metric": "mlogloss",  # Optimize probability calibration
        "tree_method": "hist",
        "device": "cuda",  # GPU Acceleration
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": 0,
        "num_class": NUM_CLASSES,
    }

    # =========================================================================
    # Training Loop Configuration
    # =========================================================================
    CV_FOLDS = 5
    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_EVAL = 100

    # =========================================================================
    # Semi-Supervised Learning (Pseudo-Labeling)
    # =========================================================================
    # Threshold for accepting a teacher prediction as a ground truth label
    PSEUDO_LABEL_THRESHOLD = 0.99

    # =========================================================================
    # Debugging / Development
    # =========================================================================
    # Set to an integer (e.g., 50000) to subsample data for fast debugging.
    # Set to None for full production training.
    DEBUG_SAMPLES = None

    @staticmethod
    def setup():
        """Creates necessary working directories."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_xgb_params(cls, overrides=None):
        """Returns a copy of XGB_PARAMS with optional overrides."""
        params = cls.XGB_PARAMS.copy()
        if overrides:
            params.update(overrides)
        return params


# Initialize directories on module import
Config.setup()
