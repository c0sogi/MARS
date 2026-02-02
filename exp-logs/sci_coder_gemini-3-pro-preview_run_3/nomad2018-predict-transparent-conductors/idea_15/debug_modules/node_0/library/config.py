import os


class Config:
    """
    Configuration class for the Steinhardt-RDF Hybrid Structural Fingerprinting pipeline.
    Contains global constants, file paths, feature extraction parameters, and model hyperparameters.
    """

    # ==========================================
    # Global Settings
    # ==========================================
    RANDOM_SEED = 42

    # Debugging: Set to an integer (e.g., 100) to process only a subset of data for testing.
    # Set to None for full production run.
    DEBUG_SAMPLE_SIZE = None

    # ==========================================
    # Directory and File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_15"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Submission Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cached Feature File Names
    TRAIN_FEATS_FILE = "train_features.parquet"
    VAL_FEATS_FILE = "val_features.parquet"
    TEST_FEATS_FILE = "test_features.parquet"

    # ==========================================
    # Feature Extraction Parameters
    # ==========================================

    # Chemical Species
    CATIONS = ["Al", "Ga", "In"]
    ANIONS = ["O"]
    ALL_ELEMENTS = ["Al", "Ga", "In", "O"]

    # Radial Distribution Function (RDF) Settings
    # Captures pairwise distance distributions
    RDF_CUTOFF = 6.0  # Max distance in Angstroms
    RDF_NUM_BINS = 60  # Number of bins for the histogram
    RDF_SIGMA = 0.2  # Width for Gaussian smearing (if applicable)

    # Steinhardt Bond Orientational Order Parameters (Q_l)
    # Captures local angular symmetry and polyhedral geometry
    STEINHARDT_L = [4, 6]  # Angular momentum indices to compute (Q4, Q6)
    STEINHARDT_CUTOFF = (
        3.0  # Neighbor cutoff in Angstroms (approx. 1st coordination shell)
    )

    # ==========================================
    # Model Hyperparameters (XGBoost)
    # ==========================================

    # Targets to predict
    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    # XGBoost Regressor Parameters
    # Optimized for generalization with shrinkage and subsampling
    XGB_MODEL_PARAMS = {
        "n_estimators": 3000,  # High number of trees
        "learning_rate": 0.01,  # Low learning rate (shrinkage)
        "max_depth": 6,  # Moderate depth to capture interactions
        "subsample": 0.7,  # Row subsampling
        "colsample_bytree": 0.6,  # Feature subsampling
        "min_child_weight": 1,
        "gamma": 0,
        "n_jobs": -1,  # Use all available cores
        "random_state": RANDOM_SEED,
        "objective": "reg:squarederror",
        "tree_method": "hist",  # Fast histogram-based algorithm
        "booster": "gbtree",
    }

    # Training Loop Settings
    EARLY_STOPPING_ROUNDS = 100  # Stop if validation score doesn't improve
    VERBOSE_EVAL = 200  # Print metrics every N rounds
