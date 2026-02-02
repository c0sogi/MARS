import os


class Config:
    """
    Central configuration for the Anisotropic-Topological Multi-Scale Fingerprinting pipeline.
    """

    # --- Directory Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_39"
    SUBMISSION_DIR = "./submission"

    # --- Metadata File Paths ---
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # --- Output Feature Paths (Caching) ---
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # --- Submission Path ---
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Reproducibility ---
    RANDOM_SEED = 42

    # --- Feature Extraction Parameters ---
    # Radial Distribution Function (RDF) settings
    RDF_CUTOFF = 6.0  # Angstroms
    RDF_NUM_BINS = 60  # Resulting in ~0.1 A bin width

    # Network Topology & Local Environment settings
    BOND_CUTOFF = 3.0  # Angstroms, used for determining neighbors for angles and BVS

    # Bond Valence Sum (BVS) Parameters
    # Approximate R0 values for Metal(III)-Oxygen bonds
    BVS_PARAMS = {
        "Al": 1.651,
        "Ga": 1.730,
        "In": 1.917,
        "O": 0.0,  # Oxygen is the anion
    }
    BVS_B_PARAM = 0.37

    # --- Model Hyperparameters (XGBoost) ---
    # Optimized for generalization with low learning rate and stochastic subsampling
    XGB_PARAMS = {
        "n_estimators": 3000,
        "learning_rate": 0.01,
        "max_depth": 6,
        "subsample": 0.65,
        "colsample_bytree": 0.65,
        "objective": "reg:squarederror",
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "tree_method": "hist",  # Efficient training
        "reg_alpha": 0.1,  # L1 regularization
        "reg_lambda": 1.0,  # L2 regularization
    }

    # --- Training Control ---
    DEBUG_MODE = False  # Set to True to run on a small subset for testing
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use in debug mode

    EARLY_STOPPING_ROUNDS = 100
    VERBOSE_EVAL = 200  # Print metrics every N rounds

    @classmethod
    def setup(cls):
        """
        Creates necessary working and submission directories if they do not exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically setup directories when config is imported
Config.setup()
