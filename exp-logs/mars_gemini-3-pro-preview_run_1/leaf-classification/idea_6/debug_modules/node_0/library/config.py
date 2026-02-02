import os


class Config:
    """
    Configuration for the Fisher-Embedded Bayesian Gaussian Process Leaf Classifier.
    """

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"
    SUBMISSION_DIR = "./submission"

    # Data paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output paths
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache paths (for deterministic processing)
    CACHE_TRAIN_PATH = os.path.join(WORKING_DIR, "X_train_processed.npy")
    CACHE_VAL_PATH = os.path.join(WORKING_DIR, "X_val_processed.npy")
    CACHE_TEST_PATH = os.path.join(WORKING_DIR, "X_test_processed.npy")

    # -------------------------------------------------------------------------
    # Dataset Parameters
    # -------------------------------------------------------------------------
    N_CLASSES = 99
    # Features: 64 margin + 64 shape + 64 texture = 192
    N_FEATURES = 192

    # Debugging / Development
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLES = 50

    # -------------------------------------------------------------------------
    # Preprocessing Hyperparameters
    # -------------------------------------------------------------------------
    # Power Transformation to enforce Gaussianity
    POWER_TRANSFORM_METHOD = "yeo-johnson"

    # Scaling to ensure numerical stability for Kernel/SVD
    SCALER_TYPE = "standard"  # StandardScaler (mean=0, std=1)

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------

    # 1. Backbone: Linear Discriminant Analysis (Fisher Embedding)
    # ------------------------------------------------------------
    LDA_SOLVER = "svd"
    # Project to N_CLASSES - 1 dimensions for maximum separation
    LDA_N_COMPONENTS = 98

    # 2. Head: Gaussian Process Classifier (Bayesian Inference)
    # ---------------------------------------------------------
    # Kernel: 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)

    # RBF Kernel Parameters
    GPC_RBF_LENGTH_SCALE = 1.0
    GPC_RBF_LENGTH_SCALE_BOUNDS = (1e-1, 1e2)

    # White Kernel Parameters (for numerical stability and noise modeling)
    GPC_NOISE_LEVEL = 1e-5
    GPC_NOISE_LEVEL_BOUNDS = (1e-10, 1e-1)

    # Optimizer Parameters
    GPC_OPTIMIZER = "fmin_l_bfgs_b"
    GPC_N_RESTARTS_OPTIMIZER = 2  # Restart optimizer to avoid local optima
    GPC_MAX_ITER_PREDICT = 100

    # Multi-class strategy
    GPC_MULTI_CLASS = "one_vs_rest"

    # -------------------------------------------------------------------------
    # Ensemble Strategy
    # -------------------------------------------------------------------------
    # Weighted average of probabilities: P_final = w1 * P_LDA + w2 * P_GPC
    WEIGHT_LDA = 0.5
    WEIGHT_GPC = 0.5

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
