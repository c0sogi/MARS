import os


class Config:
    # ==========================================
    # Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_9"
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Paths
    # ==========================================
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_meta.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_meta.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_meta.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Toggle for debugging with smaller dataset
    DEBUG_SAMPLE_SIZE = 100
    NUM_WORKERS = 12  # Based on available vCPUs

    # ==========================================
    # Image Configuration
    # ==========================================
    IMAGE_SIZE = 224
    ZOOM_CROP_RATIO = 0.6

    # ==========================================
    # Model Backbones
    # ==========================================
    # Identifiers for the 4 backbones used in the Quad-Stream Extractor
    BACKBONES = [
        "swin_large_patch4_window7_224",
        "tf_efficientnetv2_l.in21k_ft_in1k",
        "vit_large_patch14_dinov2.lvd142m",
        "openai/clip-vit-large-patch14",
    ]

    # ==========================================
    # Feature Engineering
    # ==========================================
    PCA_VARIANCE = 0.95
    METADATA_SCALE = 10.0

    # ==========================================
    # Ensemble Hyperparameters
    # ==========================================
    N_FOLDS = 5

    # Support Vector Regression
    SVR_PARAMS = {"C": 20.0, "kernel": "rbf", "gamma": "scale"}

    # K-Nearest Neighbors Regressor
    KNN_PARAMS = {"n_neighbors": 50, "weights": "distance", "n_jobs": -1}

    # ExtraTrees Regressor
    ET_PARAMS = {
        "n_estimators": 100,
        "min_samples_leaf": 5,
        "max_features": None,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": 0,
    }

    # LightGBM Regressor
    LGBM_PARAMS = {
        "n_estimators": 5000,
        "learning_rate": 0.005,
        "num_leaves": 31,
        "metric": "rmse",
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    # LightGBM Fit Parameters (Early Stopping)
    LGBM_FIT_PARAMS = {
        "eval_metric": "rmse"
        # callbacks for early stopping will be added in the training loop
    }

    # Meta-Learner (Ridge Regression)
    META_PARAMS = {"alpha": 1.0, "random_state": SEED}
