import os


class PATHS:
    """
    Defines file paths for input data, metadata, working directories, and submissions.
    """

    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory specific to Idea 34 for caching intermediate features
    WORKING_DIR = "./working/idea_34"
    SUBMISSION_DIR = "./submission"

    # Metadata files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output submission file
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)


class SIGNAL_PARAMS:
    """
    Parameters for signal processing and decomposition.
    """

    # Sampling rate inferred from data (60000 samples / 10 mins)
    SAMPLING_RATE = 100

    # Savitzky-Golay Filter settings for Trend extraction (View A)
    # Large window and quadratic order to prevent overfitting to noise
    SG_WINDOW = 51
    SG_ORDER = 2

    # Discrete Wavelet Transform settings for Texture extraction (View B)
    DWT_WAVELET = "db4"

    # Welch's Method settings for Spectral extraction (View C)
    # High nperseg for high-resolution frequency bins, especially in low freq
    WELCH_NPERSEG = 1024

    # Frequency bands for PSD integration
    FREQ_BANDS = {"low": (0.1, 3.0), "mid": (3.0, 10.0), "high": (10.0, 45.0)}


class FEATURE_PARAMS:
    """
    Parameters for feature engineering.
    """

    # Dense grid of quantiles for Trend Shape (Split-Granularity)
    QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

    # Number of non-overlapping windows for Differential Temporal Profiling
    N_TEMPORAL_SEGMENTS = 10

    # List of sensors to process
    SENSORS = [f"sensor_{i}" for i in range(1, 11)]


class MODEL_PARAMS:
    """
    Hyperparameters for the Machine Learning model.
    """

    SEED = 42
    N_FOLDS = 5

    # High-Capacity LightGBM Configuration
    LGBM_PARAMS = {
        "objective": "regression_l2",  # Mean Squared Error
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 64,  # Reduced to prevent overfitting
        "learning_rate": 0.01,  # Low learning rate for convergence
        "n_estimators": 10000,  # Large number of trees
        "feature_fraction": 0.8,  # Subsample features
        "bagging_fraction": 0.8,  # Subsample data
        "bagging_freq": 1,
        "lambda_l1": 1.0,  # L1 Regularization
        "lambda_l2": 1.0,  # L2 Regularization
        "verbosity": -1,
        "n_jobs": -1,
        "seed": 42,
        "early_stopping_rounds": 100,  # Early stopping based on validation
    }


class COMPUTE_PARAMS:
    """
    Computational resources.
    """

    N_JOBS = 12  # Available vCPUs
