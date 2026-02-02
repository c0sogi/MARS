import os
import numpy as np


class Config:
    # ==========================================================================
    # GLOBAL CONSTANTS & SEEDS
    # ==========================================================================
    RANDOM_STATE = 42
    N_JOBS = -1  # Use all available cores

    # Data Type enforcement for precision near metric floor
    # We use string 'float64' to be passed to astype()
    DTYPE = "float64"
    NP_DTYPE = np.float64

    # ==========================================================================
    # DIRECTORIES & PATHS
    # ==========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching intermediate files (Idea 37)
    WORKING_DIR = "./working/idea_37"

    # Submission directory
    SUBMISSION_DIR = "./submission"

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================================================
    # DUAL-STREAM PREPROCESSING HYPERPARAMETERS
    # ==========================================================================

    # Stream A: Parametric Gaussian Anchors
    # Uses PowerTransformer with Yeo-Johnson
    PT_METHOD = "yeo-johnson"
    PT_STANDARDIZE = True

    # Stream B: Constrained Non-Parametric Experts
    # Uses QuantileTransformer with strict constraints to prevent overfitting
    QT_N_QUANTILES = 50  # Constrained to ~7% of N to avoid memorization
    QT_OUTPUT_DIST = "normal"

    # ==========================================================================
    # MODEL HYPERPARAMETERS
    # ==========================================================================

    # LDA Configuration
    # Library of shrinkage estimators for the Generative Ensemble
    # 'auto' corresponds to the Ledoit-Wolf lemma (sklearn default),
    # but we specifically want OAS and Fixed values per the idea description.
    # We will handle 'oas' as a special string flag in the implementation.
    LDA_SHRINKAGE_VALUES = [0.0001, 0.001, 0.01, 0.1]

    # Logistic Regression Configuration
    # Used as the Discriminative Backup
    LR_CV_FOLDS = 5
    LR_MAX_ITER = 10000
    LR_SOLVER = "lbfgs"  # Robust default for multiclass
    LR_SCORING = "neg_log_loss"
    # Dense Logarithmic Grid is handled by LogisticRegressionCV automatically (Cs=10 default)
    LR_CS = 10

    # ==========================================================================
    # CACHE KEYS
    # ==========================================================================
    # Filenames for cached numpy arrays/parquets
    CACHE_TRAIN_FEATURES = "X_train_processed.parquet"
    CACHE_VAL_FEATURES = "X_val_processed.parquet"
    CACHE_TEST_FEATURES = "X_test_processed.parquet"
    CACHE_TRAIN_TARGETS = "y_train.npy"
    CACHE_VAL_TARGETS = "y_val.npy"

    @classmethod
    def setup(cls):
        """
        Create necessary directories if they don't exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
Config.setup()
