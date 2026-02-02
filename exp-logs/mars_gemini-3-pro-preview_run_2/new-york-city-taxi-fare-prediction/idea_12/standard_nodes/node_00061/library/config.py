import os
import numpy as np
import random


class Config:
    """
    Configuration class for the Taxi Fare Prediction task using
    Disjoint Background-Foreground Gradient Boosting.
    """

    def __init__(self, debug=False):
        # Global Seed
        self.SEED = 42

        # ---------------------------------------------------------------------
        # File Paths
        # ---------------------------------------------------------------------
        self.INPUT_DIR = "./input"
        self.METADATA_DIR = "./metadata"

        # Input Data Paths (from Metadata)
        self.TRAIN_DATA_PATH = os.path.join(self.METADATA_DIR, "train.parquet")
        self.VAL_DATA_PATH = os.path.join(self.METADATA_DIR, "val.parquet")
        self.TEST_DATA_PATH = os.path.join(self.METADATA_DIR, "test.parquet")
        self.SAMPLE_SUBMISSION_PATH = os.path.join(
            self.INPUT_DIR, "sample_submission.csv"
        )

        # Working Directory for Cache and Intermediate Files
        # Segregate cache to prevent debug artifacts from contaminating full runs
        suffix = "_debug" if debug else "_full"
        self.WORKING_DIR = f"./working/idea_12{suffix}"
        self.SUBMISSION_DIR = "./submission"
        self.FINAL_SUBMISSION_PATH = os.path.join(self.SUBMISSION_DIR, "submission.csv")

        # ---------------------------------------------------------------------
        # K-Fold Target Encoding Strategy Constants
        # ---------------------------------------------------------------------
        self.N_FOLDS = 5

        if debug:
            self.TRAIN_SUBSAMPLE_SIZE = 100_000
            self.VAL_SUBSET_SIZE = 50_000
        else:
            # Use a large subsample for training the tree, but use FULL data for encoding
            self.TRAIN_SUBSAMPLE_SIZE = 6_000_000
            self.VAL_SUBSET_SIZE = None

        # ---------------------------------------------------------------------
        # Data Hygiene Thresholds
        # ---------------------------------------------------------------------
        # Strict Hygiene for Training Data (used for both Stats and Learning)
        # Removing outliers is critical for RMSE stability with Target Encoding
        self.MIN_FARE = 2.50
        self.MAX_FARE = 500.00
        self.MAX_FARE_PER_KM = 20.00

        # ---------------------------------------------------------------------
        # Feature Engineering Parameters
        # ---------------------------------------------------------------------
        # Simulating Geohash granularities using decimal place rounding
        # 3 decimal places ~ 110m (Fine / Geohash 7 approx)
        # 2 decimal places ~ 1.1km (Coarse / Geohash 6 approx)
        self.PRECISION_FINE = 3
        self.PRECISION_COARSE = 2

        # NYC Bounding Box for Post-Processing/Sanity Checks
        self.NYC_LAT_MIN = 40.5
        self.NYC_LAT_MAX = 41.0
        self.NYC_LON_MIN = -74.3
        self.NYC_LON_MAX = -73.7

        # ---------------------------------------------------------------------
        # Model Hyperparameters (XGBoost)
        # ---------------------------------------------------------------------
        self.XGB_PARAMS = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "learning_rate": 0.05,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": 12,
            "tree_method": "hist",  # Efficient for large data
            "device": "cuda",  # Use A100 GPU
            "random_state": self.SEED,
            "n_estimators": 5000,  # High cap, controlled by early stopping
            "early_stopping_rounds": 50,
        }

        if debug:
            self.XGB_PARAMS["n_estimators"] = 100
            self.XGB_PARAMS["tree_method"] = (
                "hist"  # Fallback if GPU has issues in debug, but prefer hist
            )

    def setup_dirs(self):
        """Creates necessary working and submission directories."""
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)
        print(f"Directories ensured: {self.WORKING_DIR}, {self.SUBMISSION_DIR}")

    def set_seed(self):
        """Sets random seeds for reproducibility."""
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        # Torch is not strictly required by the prompt's model choice (XGBoost),
        # but good practice if used later.
        try:
            import torch

            torch.manual_seed(self.SEED)
            torch.cuda.manual_seed_all(self.SEED)
        except ImportError:
            pass
        print(f"Random seed set to {self.SEED}")

    def get_cache_path(self, filename):
        """Returns the full path for a cached file in the working directory."""
        return os.path.join(self.WORKING_DIR, filename)


# Instantiate a default config object for easy import
cfg = Config()
