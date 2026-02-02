import os


class Config:
    """
    Global configuration for the Physics-Informed Feature Engineering pipeline.
    """

    # --- Global Settings ---
    RANDOM_SEED = 42

    # --- Directories ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_9"
    SUBMISSION_DIR = "./submission"

    # --- Input Files ---
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # --- Output Files (Caching) ---
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Feature Engineering Constants ---
    # Atomic properties for Chemical Disorder and Electrostatic features
    # EN: Pauling Electronegativity
    # R: Shannon Ionic Radius (Angstroms) for coordination number VI (typical for these oxides)
    # Z: Atomic Number
    ATOMIC_PROPS = {
        "Al": {"EN": 1.61, "R": 0.54, "Z": 13},
        "Ga": {"EN": 1.81, "R": 0.62, "Z": 31},
        "In": {"EN": 1.78, "R": 0.80, "Z": 49},
        "O": {"EN": 3.44, "R": 1.40, "Z": 8},
    }

    # --- Model Hyperparameters ---
    # Optimized for generalization on physical features
    # Low learning rate + high estimators (Shrinkage)
    # Subsampling to reduce reliance on dominant features like density
    XGB_PARAMS = {
        "n_estimators": 3000,
        "learning_rate": 0.01,
        "max_depth": 6,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "objective": "reg:squarederror",
        "tree_method": "hist",
    }

    # --- Training Configuration ---
    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    # Debugging
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use when DEBUG is True

    @classmethod
    def setup(cls):
        """Ensures that working and submission directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
