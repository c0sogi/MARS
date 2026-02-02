import os
import torch


class Config:
    # Global Seed for reproducibility
    SEED = 42

    class Paths:
        # Input Directories
        INPUT_DIR = "./input"
        METADATA_DIR = "./metadata"

        # Working Directory for this specific idea (Idea 12)
        # All cached files and models should be saved here
        WORKING_DIR = "./working/idea_12"

        # Subdirectories for organization
        CACHE_DIR = os.path.join(WORKING_DIR, "cache")
        MODEL_OUTPUT_DIR = os.path.join(WORKING_DIR, "fine_tuned_mpnet")
        SUBMISSION_DIR = "./submission"

        # Metadata Files
        TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
        VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
        TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

        # Submission File
        SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

        @classmethod
        def setup_dirs(cls):
            """Creates necessary output directories."""
            os.makedirs(cls.WORKING_DIR, exist_ok=True)
            os.makedirs(cls.CACHE_DIR, exist_ok=True)
            os.makedirs(cls.MODEL_OUTPUT_DIR, exist_ok=True)
            os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    class Model:
        # Backbone architecture
        BASE_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

        # Sequence length configuration
        # MPNet supports up to 512. We use 128 for efficiency while retaining context.
        MAX_SEQ_LEN = 128

        # Compute device
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    class Features:
        # Structural Heatmap Configuration
        HEATMAP_BINS = 20  # Fixed length for interpolated similarity vector

        # Smoothing Configuration
        SMOOTHING_WINDOW = 3  # Window size for 1D convolution

        # Context Features
        USE_CONTEXT_FEATURES = True  # Include n_code and md_len

    class Training:
        # --- Stage 1: Contrastive Fine-Tuning ---

        # Dataset size constraints
        # "Construct pairs from 40,000 notebooks"
        NUM_NOTEBOOKS_FINE_TUNE = 40000

        # Training hyperparameters
        FINE_TUNE_EPOCHS = 1
        FINE_TUNE_BATCH_SIZE = 32  # Adjusted for A100 40GB with MPNet
        FINE_TUNE_LR = 2e-5

        # Data Loading
        NUM_WORKERS = 4

        # --- Stage 2: Regression (LightGBM) ---

        # Dataset size constraints
        # "Utilize the Full Training Dataset"
        # Setting to None implies using all available data in the metadata file
        NUM_NOTEBOOKS_LGBM = None

        # LightGBM Hyperparameters
        LGBM_PARAMS = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "n_estimators": 2000,
            "num_leaves": 31,
            "max_depth": -1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbosity": -1,  # Silent mode
            "n_jobs": 12,  # Utilize available vCPUs
            "random_state": 42,
        }

        # Early stopping for LGBM
        LGBM_EARLY_STOPPING_ROUNDS = 50


# Ensure directories exist upon import
Config.Paths.setup_dirs()
