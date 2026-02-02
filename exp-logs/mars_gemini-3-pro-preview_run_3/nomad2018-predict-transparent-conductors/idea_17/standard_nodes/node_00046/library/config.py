import os


class Config:
    """
    Global configuration for the Hybrid Geometric-Sublattice Fingerprinting pipeline.
    """

    # -------------------------------------------------------------------------
    # Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_18"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    # Metadata files (pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cached Feature Paths (Parquet format)
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # Final Submission Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Global Settings
    # -------------------------------------------------------------------------
    RANDOM_SEED = 42

    # Debugging / Development
    # Set SAMPLE_SIZE to a small integer (e.g., 100) to run a quick test on a subset.
    # Set to None for full run.
    SAMPLE_SIZE = None

    # -------------------------------------------------------------------------
    # Feature Extraction Hyperparameters
    # -------------------------------------------------------------------------
    # Radial Distribution Function (RDF)
    RDF_CUTOFF = 6.0
    RDF_NUM_BINS = 64

    # Chemically-Resolved Local Environment Moments (CR-LEM)
    LEM_CUTOFF = 3.0

    # -------------------------------------------------------------------------
    # Model Hyperparameters (XGBoost)
    # -------------------------------------------------------------------------
    # Cite solution_lesson_node_00045: Optimized for geometric features (RDF+LEM)
    # Relaxed regularization compared to GNN-hybrid config to prevent underfitting
    XGB_PARAMS = {
        "n_estimators": 5000,
        "learning_rate": 0.01,
        "max_depth": 8,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 1,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }

    # Training Control
    EARLY_STOPPING_ROUNDS = 150
    VERBOSE_EVAL = 200

    # Target Variables
    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
