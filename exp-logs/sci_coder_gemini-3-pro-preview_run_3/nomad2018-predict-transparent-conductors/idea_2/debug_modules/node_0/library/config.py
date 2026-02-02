import os


class Config:
    """
    Configuration constants and settings for the Hybrid GNN-GBDT pipeline.
    """

    # --- General Settings ---
    RANDOM_SEED = 42

    # --- File Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # Input Metadata
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Caching Paths (Parquet files for deterministic processing)
    # These store the combined feature matrix (GNN embeddings + Tabular + Physical)
    TRAIN_FEATURES_CACHE = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_CACHE = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_CACHE = os.path.join(WORKING_DIR, "test_features.parquet")

    # --- Data Configuration ---
    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    # Apply log(1+x) transformation to targets to align with RMSLE metric
    LOG_TRANSFORM_TARGETS = True

    # Debugging: Set to an integer (e.g., 50) to process only a subset of data.
    # Set to None for full training.
    DEBUG_SAMPLE_SIZE = None

    # --- Feature Extraction (GNN) ---
    # Using M3GNet from matgl as the backbone for structural embeddings
    MATGL_MODEL_NAME = "M3GNet-MP-2021.2.8-PES"
    GNN_BATCH_SIZE = 16  # Batch size for graph inference

    # --- Model Hyperparameters (XGBoost) ---
    # Strategy: Shrinkage and Subsampling for robustness
    XGB_PARAMS = {
        "n_estimators": 2500,  # High number of trees
        "learning_rate": 0.01,  # Low learning rate (shrinkage)
        "max_depth": 7,  # Moderate depth to capture interactions
        "subsample": 0.7,  # Row subsampling to prevent overfitting
        "colsample_bytree": 0.7,  # Column subsampling
        "min_child_weight": 5,  # Regularization
        "gamma": 0.1,  # Minimum loss reduction
        "objective": "reg:squarederror",
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "tree_method": "hist",  # Efficient training method
    }

    # Training Loop Settings
    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_EVAL = 100

    @classmethod
    def initialize(cls):
        """
        Creates the necessary working and submission directories if they do not exist.
        This should be called when the config is imported.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Ensure directories exist upon import
Config.initialize()
