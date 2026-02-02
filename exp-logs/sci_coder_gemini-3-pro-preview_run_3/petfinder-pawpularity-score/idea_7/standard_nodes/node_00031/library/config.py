import os
import torch


class Config:
    # ==========================================
    # 1. Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    # Metadata Paths (Generated in previous steps)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_meta.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_meta.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_meta.csv")

    # Submission Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # ==========================================
    # 2. Global Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 100

    # Compute
    NUM_WORKERS = 12
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # 3. Data Preprocessing
    # ==========================================
    IMG_SIZE = 224
    BATCH_SIZE = 32

    # Spatial Pyramid Pooling (SPP)
    # Levels: 1x1 (Global Average) and 2x2 (Quadrant Average)
    # Output vector size multiplier: 1 + 4 = 5x
    SPP_LEVELS = [1, 2]

    # PCA Compression
    # Retain 95% of variance to handle high dimensionality from SPP
    PCA_VARIANCE = 0.95

    # Metadata Features
    METADATA_COLS = [
        "Focus",
        "Eyes",
        "Face",
        "Near",
        "Action",
        "Accessory",
        "Group",
        "Collage",
        "Human",
        "Occlusion",
        "Info",
        "Blur",
    ]
    # Scaling factor for binary metadata to ensure impact in distance-based models (SVR)
    METADATA_SCALE = 10.0

    # ==========================================
    # 4. Feature Extraction Backbones
    # ==========================================
    # Dictionary mapping friendly names to model identifiers
    BACKBONES = {
        "swin": "swin_large_patch4_window7_224",  # timm: Global composition
        "efficientnet": "tf_efficientnetv2_l.in21k_ft_in1k",  # timm: Texture/Quality
        "dinov2": "vit_large_patch14_dinov2.lvd142m",  # timm: Geometric correspondence
        "clip": "openai/clip-vit-large-patch14",  # transformers: Semantic/Vibe
    }

    # ==========================================
    # 5. Ensemble Model Hyperparameters
    # ==========================================

    # Level 1: Support Vector Regression
    SVR_PARAMS = {
        "C": 20.0,
        "kernel": "rbf",
        "epsilon": 0.1,
        "gamma": "scale",
        "cache_size": 2000,
    }

    # Level 1: LightGBM Regressor
    LGBM_PARAMS = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.01,
        "n_estimators": 2000,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": SEED,
    }

    # Level 1: ExtraTrees Regressor
    EXTRATREES_PARAMS = {
        "n_estimators": 1000,
        "max_depth": None,
        "max_features": None,  # No feature subsampling (Use all PCA components)
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "bootstrap": False,
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": 0,
    }

    # Level 2: Meta Learner (Linear Regression)
    META_LEARNER_PARAMS = {"fit_intercept": True, "copy_X": True, "n_jobs": -1}

    # ==========================================
    # 6. Training & Validation
    # ==========================================
    N_FOLDS = 5
    LGBM_EARLY_STOPPING_ROUNDS = 50

    # ==========================================
    # 7. Setup Logic
    # ==========================================
    @staticmethod
    def setup():
        """Creates necessary working directories."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# Execute setup on import
Config.setup()
