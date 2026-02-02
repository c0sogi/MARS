import os
import torch


class Config:
    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./working/submission"

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
    VAL_PATH = os.path.join(
        METADATA_DIR, "val.parquet"
    )  # Available but we use full CV on train usually
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache paths for processed data
    CACHE_TRAIN_X = os.path.join(WORKING_DIR, "X_train_processed.parquet")
    CACHE_TRAIN_Y = os.path.join(WORKING_DIR, "y_train_processed.npy")
    CACHE_TEST_X = os.path.join(WORKING_DIR, "X_test_processed.parquet")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")

    # =========================================================================
    # Data Definitions
    # =========================================================================
    ID_COL = "Id"
    TARGET_COL = "Cover_Type"

    # Continuous / Numerical Features
    NUMERIC_COLS = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]

    # Binary / Categorical Features
    # Wilderness Areas (4)
    WILDERNESS_COLS = [f"Wilderness_Area{i}" for i in range(1, 5)]
    # Soil Types (40)
    SOIL_COLS = [f"Soil_Type{i}" for i in range(1, 41)]

    BINARY_COLS = WILDERNESS_COLS + SOIL_COLS

    # All input features
    ALL_FEATURES = NUMERIC_COLS + BINARY_COLS

    # Class definitions
    # Classes are [1, 2, 3, 4, 6, 7] based on EDA.
    # We will map them to [0, 1, 2, 3, 4, 5] for training and map back for submission.
    # Note: Class 5 is missing.
    ORIGINAL_LABELS = [1, 2, 3, 4, 6, 7]
    NUM_CLASSES = len(ORIGINAL_LABELS)

    # =========================================================================
    # Global Training Settings
    # =========================================================================
    SEED = 42
    N_FOLDS = 5
    USE_GPU = torch.cuda.is_available()
    DEVICE = "cuda" if USE_GPU else "cpu"

    # =========================================================================
    # XGBoost Hyperparameters
    # =========================================================================
    XGB_PARAMS = {
        "n_estimators": 3000,
        "learning_rate": 0.05,
        "max_depth": 10,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "objective": "multi:softprob",
        "num_class": NUM_CLASSES,
        "device": "cuda" if USE_GPU else "cpu",
        "tree_method": "hist",  # 'hist' is efficient on GPU with device='cuda' in XGB 3.x
        "eval_metric": "mlogloss",
        "random_state": SEED,
        "n_jobs": -1,
        "early_stopping_rounds": 100,
        "verbose": 0,
    }

    # =========================================================================
    # Neural Network Hyperparameters
    # =========================================================================
    NN_PARAMS = {
        "batch_size": 2048,
        "epochs": 30,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "hidden_layers": [512, 256, 128],
        "dropout": 0.2,
        "scheduler_patience": 3,
        "early_stopping_patience": 6,
        "use_batch_norm": True,
    }

    # =========================================================================
    # Feature Engineering Flags
    # =========================================================================
    # Generate interactions between Wilderness Areas and Distance/Elevation features
    ADD_INTERACTIONS = True
    # Apply QuantileTransformer (GaussRank) to numeric features for NN
    SCALE_NUMERICS_NN = True
