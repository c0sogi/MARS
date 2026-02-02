import os


class Config:
    # ==========================================
    # 1. Global Configuration
    # ==========================================
    SEED = 42
    N_FOLDS = 5
    NUM_WORKERS = 4

    # ==========================================
    # 2. Directories
    # ==========================================
    INPUT_DIR = "./input"
    # Metadata is already generated in ./metadata
    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_meta.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_meta.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_meta.csv")

    # Working directory for this specific idea (Idea 8)
    WORKING_DIR = "./working/idea_8"
    SUBMISSION_DIR = "./submission"

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # 3. Data Parameters
    # ==========================================
    IMAGE_SIZE = (224, 224)
    # Batch size for feature extraction (A100 can handle 32-64 for large models)
    BATCH_SIZE = 32

    # Metadata Columns (Binary Features)
    # Note: 'Subject Focus' is the column name in the CSV based on analysis,
    # though description sometimes refers to it as 'Focus'.
    METADATA_COLS = [
        "Subject Focus",
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

    TARGET_COL = "Pawpularity"

    # ==========================================
    # 4. Backbone Configuration
    # ==========================================
    # List of backbones to use for the Quad-Stream approach.
    # source: 'timm' or 'transformers' (huggingface)
    BACKBONES = [
        {
            "name": "microsoft/swin-large-patch4-window7-224",
            "source": "transformers",
            "type": "swin",
        },
        {"name": "tf_efficientnetv2_l.in21k_ft_in1k", "source": "timm", "type": "cnn"},
        {"name": "facebook/dinov2-large", "source": "transformers", "type": "vit"},
        {
            "name": "openai/clip-vit-large-patch14",
            "source": "transformers",
            "type": "clip",
        },
    ]

    # ==========================================
    # 5. Feature Engineering Hyperparameters
    # ==========================================
    # Dual-Pooling is implied by the architecture (Avg + Max)

    # PCA Compression
    PCA_VARIANCE = 0.95

    # Interaction Engineering
    METADATA_SCALING = 10.0
    INTERACTION_TOP_K = 3  # Number of top PCA components to interact with metadata

    # ==========================================
    # 6. Model Hyperparameters
    # ==========================================
    # Level 1: Support Vector Regression
    SVR_PARAMS = {
        "kernel": "rbf",
        "C": 20.0,  # High regularization for dense feature space
        "gamma": "scale",
        "epsilon": 0.1,
    }

    # Level 1: ExtraTrees Regressor
    EXTRATREES_PARAMS = {
        "n_estimators": 100,
        "max_features": None,  # Evaluate all features at split
        "min_samples_leaf": 1,
        "random_state": SEED,
        "n_jobs": -1,
    }

    # Level 1: LightGBM
    LGBM_PARAMS = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.01,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_estimators": 1000,
        "verbose": -1,
        "random_state": SEED,
        "n_jobs": -1,
    }

    # Training settings
    EARLY_STOPPING_ROUNDS = 50
