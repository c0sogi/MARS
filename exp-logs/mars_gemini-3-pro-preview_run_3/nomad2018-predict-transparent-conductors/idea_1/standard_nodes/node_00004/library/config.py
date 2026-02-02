import os
import numpy as np


class Config:
    # ==========================================
    # Directories and Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for caching intermediate files (e.g., extracted features)
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"

    # Metadata file paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache file paths (parquet format preferred)
    # Distinct paths for geometry features (raw) and processed features (engineered)
    TRAIN_GEO_CACHE = os.path.join(WORKING_DIR, "train_geometry.parquet")
    VAL_GEO_CACHE = os.path.join(WORKING_DIR, "val_geometry.parquet")
    TEST_GEO_CACHE = os.path.join(WORKING_DIR, "test_geometry.parquet")

    TRAIN_PROCESSED_CACHE = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_PROCESSED_CACHE = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_PROCESSED_CACHE = os.path.join(WORKING_DIR, "test_processed.parquet")

    # ==========================================
    # Column Definitions
    # ==========================================
    ID_COL = "id"
    FILE_PATH_COL = "file_path"

    # Target variables
    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    # Categorical features to be one-hot encoded
    CAT_COLS = ["spacegroup"]

    # Numerical features present in the CSV files
    NUM_COLS = [
        "number_of_total_atoms",
        "percent_atom_al",
        "percent_atom_ga",
        "percent_atom_in",
        "lattice_vector_1_ang",
        "lattice_vector_2_ang",
        "lattice_vector_3_ang",
        "lattice_angle_alpha_degree",
        "lattice_angle_beta_degree",
        "lattice_angle_gamma_degree",
    ]

    # Features extracted from geometry.xyz files
    GEO_COLS = ["geo_volume", "geo_density", "geo_num_atoms"]

    # ==========================================
    # Model Hyperparameters & Pipeline Settings
    # ==========================================
    RANDOM_SEED = 42

    # Data Processing
    # Set to a small number (e.g., 100) for debugging, or None for full dataset
    DEBUG_SAMPLE_SIZE = None

    # Feature Engineering
    # Polynomial features removed as XGBoost handles non-linearities (Cite solution_lesson_node_00003)

    # Target Transformation
    # Apply log(1+x) to targets before training and exp(x)-1 to predictions (Cite solution_lesson_node_00002)
    APPLY_LOG_TARGET = True

    # XGBoost Hyperparameters
    XGB_PARAMS = {
        "n_estimators": 2500,
        "learning_rate": 0.015,
        "max_depth": 6,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "random_state": 42,
    }

    @classmethod
    def setup(cls):
        """
        Ensures that working and submission directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Directories ensured: {cls.WORKING_DIR}, {cls.SUBMISSION_DIR}")


# Perform setup upon import to ensure directories are ready
Config.setup()
