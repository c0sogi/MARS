import os


class Config:
    """
    Centralized configuration for the Scalar Coupling Prediction task.
    Includes file paths, data column definitions, and model hyperparameters.
    """

    # Global Random Seed for reproducibility
    RANDOM_SEED = 42

    # -------------------------------------------------------------------------
    # Directory Setup
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    # Metadata files (Pre-split)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Raw Data files
    STRUCTURES_PATH = os.path.join(INPUT_DIR, "structures.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache files for processed data (Parquet format)
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")

    # -------------------------------------------------------------------------
    # Column Definitions
    # -------------------------------------------------------------------------
    # Identifiers and Targets
    ID_COL = "id"
    TARGET_COL = "scalar_coupling_constant"
    MOLECULE_COL = "molecule_name"

    # Raw Input Columns
    ATOM_INDEX_0_COL = "atom_index_0"
    ATOM_INDEX_1_COL = "atom_index_1"
    TYPE_COL = "type"

    # Generated Feature Columns
    DIST_COL = "dist"
    DIST_INV_COL = "dist_inv"  # 1/d
    DIST_INV2_COL = "dist_inv2"  # 1/d^2
    DIST_INV3_COL = "dist_inv3"  # 1/d^3
    TYPE_ENC_COL = "type_enc"  # Label Encoded Coupling Type
    ATOM_0_ENC_COL = "atom_0_enc"  # Label Encoded Atom Type 0
    ATOM_1_ENC_COL = "atom_1_enc"  # Label Encoded Atom Type 1

    # List of features to be used for training
    FEATURES = [
        "atom_index_0",
        "atom_index_1",
        "type_enc",
        "atom_0_enc",
        "atom_1_enc",
        "dist",
        "dist_inv",
        "dist_inv2",
        "dist_inv3",
        "en_0",
        "en_1",
        "en_diff",
        "n_bonds_0",
        "n_bonds_1",
        "min_dist_neigh_0",
        "min_dist_neigh_1",
        "mean_dist_neigh_0",
        "mean_dist_neigh_1",
        # Neighbor Type Counts (Bag of Atoms)
        "n_H_0",
        "n_C_0",
        "n_N_0",
        "n_O_0",
        "n_F_0",
        "n_H_1",
        "n_C_1",
        "n_N_1",
        "n_O_1",
        "n_F_1",
    ]

    # -------------------------------------------------------------------------
    # Model Hyperparameters (XGBoost)
    # -------------------------------------------------------------------------
    # Parameters for the XGBRegressor constructor
    XGB_PARAMS = {
        "n_estimators": 10000,  # High limit, controlled by early stopping
        "learning_rate": 0.1,
        "max_depth": 9,  # Deeper trees for complex interactions
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "tree_method": "hist",  # Efficient histogram-based algorithm
        "device": "cuda",  # Use NVIDIA A100 GPU
        "n_jobs": 12,  # Number of CPU threads
        "random_state": RANDOM_SEED,
        "early_stopping_rounds": 100,
    }

    # Parameters for the fit method
    XGB_FIT_PARAMS = {"verbose": 100}
