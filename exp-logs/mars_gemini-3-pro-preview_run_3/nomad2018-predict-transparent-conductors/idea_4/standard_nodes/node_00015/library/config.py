import os


class Config:
    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache Files (Parquet)
    TRAIN_FEATS_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATS_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATS_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # Submission File
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Data Configuration ---
    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
    ID_COL = "id"
    FILE_PATH_COL = "file_path"

    # --- Model Hyperparameters ---
    # MatGL / M3GNet Settings
    # Using a pre-trained potential model from MatGL
    MATGL_MODEL_NAME = "M3GNet-MP-2021.2.8-PES"

    # XGBoost Settings (tuned for generalization per strategy)
    XGB_PARAMS = {
        "n_estimators": 2500,
        "learning_rate": 0.01,
        "max_depth": 6,
        "subsample": 0.65,
        "colsample_bytree": 0.65,
        "n_jobs": -1,
        "random_state": 42,
        "tree_method": "hist",  # Faster training
    }

    # --- Reproducibility ---
    RANDOM_SEED = 42

    @classmethod
    def setup(cls):
        """Creates necessary directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories when module is imported
Config.setup()
