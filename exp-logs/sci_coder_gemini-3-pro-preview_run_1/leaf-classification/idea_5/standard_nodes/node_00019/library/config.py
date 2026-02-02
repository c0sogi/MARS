import os


class Config:
    """
    Configuration class for the Bayesian Fisher-Gaussian Process Pipeline.
    Defines file paths, data constants, and model hyperparameters.
    """

    # ==========================================
    # Directories and Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Cache directory for idea_5 (Bayesian Fisher-Gaussian Process)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_5")

    # Submission directory
    SUBMISSION_DIR = "./submission"

    # Ensure necessary writeable directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Dataset File Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output File Paths
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Definitions
    # ==========================================
    ID_COL = "id"
    TARGET_COL = "species"
    FILE_PATH_COL = "file_path"

    # Feature definitions
    # The dataset has 3 groups of 64 features each
    FEATURE_GROUPS = ["margin", "shape", "texture"]
    FEATURES_PER_GROUP = 64
    TOTAL_FEATURES = 192

    # ==========================================
    # Global Settings
    # ==========================================
    RANDOM_SEED = 42

    # ==========================================
    # Model Hyperparameters
    # ==========================================

    # 1. Preprocessing
    # Yeo-Johnson is strictly retained to enforce multivariate Gaussian distribution
    PREPROCESSING = {
        "method": "yeo-johnson",
        "standardize": True,  # Apply StandardScaler after PowerTransformer
    }

    # 2. Backbone: Linear Discriminant Analysis (LDA)
    # Used as a supervised dimensionality reduction transformer and baseline classifier
    LDA_PARAMS = {
        "solver": "svd",  # SVD is preferred for stability
        "store_covariance": True,  # Required for some LDA properties
        "tol": 1e-4,
    }

    # 3. Head: Gaussian Process Classifier (GPC)
    # Trained on the projected Fisher features
    GPC_PARAMS = {
        "n_restarts_optimizer": 2,  # To avoid local optima during kernel hyperparam tuning
        "max_iter_predict": 100,
        "copy_X_train": False,  # Memory optimization
        "random_state": RANDOM_SEED,
        "n_jobs": -1,  # Use all available cores
    }

    # GPC Kernel Configuration
    # We use a compound kernel: RBF (smoothness) + WhiteKernel (noise/jitter)
    KERNEL_PARAMS = {
        "rbf_length_scale": 1.0,
        "rbf_length_scale_bounds": (1e-2, 1e2),
        "white_noise_level": 1e-5,
        "white_noise_level_bounds": (1e-10, 1e-1),
    }

    # 4. Ensemble Strategy
    # Hybrid Generative-Discriminative ensemble
    # Weighted average of LDA probabilities and GPC probabilities
    ENSEMBLE_WEIGHTS = {"lda": 0.5, "gpc": 0.5}
