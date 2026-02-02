import os
import hashlib
import json


class Config:
    # ==========================================
    # General Configuration
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use when DEBUG is True
    NUM_WORKERS = 4  # Number of DataLoader workers
    DEVICE = "cuda"  # 'cuda' or 'cpu'

    # ==========================================
    # Data Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory specific to Idea 3 (Stacked Hybrid-Feature Ensemble)
    WORKING_DIR = "./working/idea_3"

    # Metadata files (generated previously)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_meta.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_meta.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_meta.csv")

    # Output submission path
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Image Processing & Feature Extraction
    # ==========================================
    IMAGE_SIZE = 224
    BATCH_SIZE = 32

    # Normalization constants (ImageNet defaults)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Dual-Backbone Architecture
    # Using timm library naming conventions
    BACKBONES = ["swin_base_patch4_window7_224", "tf_efficientnetv2_m"]

    # Test-Time Augmentation (TTA)
    # If True, processes original and horizontally flipped images, averaging features.
    USE_TTA = True

    # ==========================================
    # Stacking Ensemble Configuration
    # ==========================================
    N_FOLDS = 5

    # Level 1 Model Hyperparameters

    # 1. Support Vector Regression (SVR)
    SVR_PARAMS = {"kernel": "rbf", "C": 10.0, "gamma": "scale", "epsilon": 0.1}
    # PCA for SVR dimensionality reduction (to speed up training)
    SVR_PCA_COMPONENTS = 0.95  # Keep components explaining 95% variance

    # 2. LightGBM Regressor
    LGBM_PARAMS = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.01,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_estimators": 2000,
        "early_stopping_rounds": 100,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": SEED,
    }

    # 3. Ridge Regression
    RIDGE_ALPHA = 1.0

    # ==========================================
    # Methods
    # ==========================================
    @classmethod
    def get_config_hash(cls):
        """
        Generates a unique MD5 hash based on the configuration parameters that
        affect feature extraction (Backbones, Image Size, TTA).

        This hash is used to version cached feature files. If the pipeline logic
        changes (e.g., different backbones or TTA setting), the hash changes,
        forcing a re-computation of features rather than loading stale cache.
        """
        config_dict = {
            "image_size": cls.IMAGE_SIZE,
            "backbones": sorted(cls.BACKBONES),
            "use_tta": cls.USE_TTA,
            "mean": cls.IMAGENET_MEAN,
            "std": cls.IMAGENET_STD,
        }

        # Serialize to JSON string with sorting keys for determinism
        config_str = json.dumps(config_dict, sort_keys=True)

        # Generate MD5 hash
        return hashlib.md5(config_str.encode("utf-8")).hexdigest()

    @classmethod
    def setup(cls):
        """
        Creates necessary directories for the working environment.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
