import os
import numpy as np


class Config:
    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_33"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Raw Data Files
    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")
    TRAIN_HELMETS_PATH = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
    TEST_HELMETS_PATH = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # =========================================================================
    # Global Configuration
    # =========================================================================
    SEED = 42
    N_JOBS = 12  # Number of vCPUs available

    # Sampling Strategy
    # Retain 100% positives, subsample negatives to this ratio (Negative : Positive)
    UNDERSAMPLE_RATIO = 10.0

    # =========================================================================
    # Feature Engineering Configuration
    # =========================================================================

    # Temporal Window: Dense Immediate Lags + Sparse Outer Lags
    # Captures Approach -> Impact -> Reaction
    LAGS = [-15, -8, -4, -2, -1, 0, 1, 2, 4, 8, 15]

    # Scale-Aligned Consistency
    # Heuristic factor to align physical closure rate (yards/s) with visual looming rate (pixels/s)
    # Used for the composite feature: (Raw_Closure_Rate * GLOBAL_SCALE_FACTOR) - Visual_Looming_Rate
    GLOBAL_SCALE_FACTOR = 15.0

    # Stream A (Interaction) Feature Groups
    # Applied when player2 != 'G'
    STREAM_A_FEATURES = {
        "relational": ["distance", "closure_rate"],  # Calculated between p1 and p2
        "consistency": [
            "consistency_composite",
            "raw_closure_rate",
            "visual_looming_rate",
        ],
        "visual_consensus": ["max_iou", "min_iou", "iou_diff"],  # At sparse lags
        "system_energy": ["speed", "acceleration"],  # For both p1 and p2
    }

    # Stream B (Impact) Feature Groups
    # Applied when player2 == 'G'
    STREAM_B_FEATURES = {
        "field_centric": [
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "direction",
            "orientation",
        ],
        "ego_dynamics": ["v_surge", "v_sway", "rotational_energy", "ego_jerk"],
    }

    # =========================================================================
    # Model Hyperparameters (Asymmetric)
    # =========================================================================

    # Common XGBoost Params
    COMMON_XGB_PARAMS = {
        "booster": "gbtree",
        "tree_method": "gpu_hist",  # Use A100 GPU
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": SEED,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    # Stream A: Interaction Model (Consistency & Energy)
    # Standard depth for complex cross-modal interactions
    STREAM_A_PARAMS = COMMON_XGB_PARAMS.copy()
    STREAM_A_PARAMS.update(
        {
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 2000,
            "early_stopping_rounds": 50,
        }
    )

    # Stream B: Impact Model (Hybrid Context & Dynamics)
    # Shallow depth and conservative learning rate to prevent overfitting sensor noise
    STREAM_B_PARAMS = COMMON_XGB_PARAMS.copy()
    STREAM_B_PARAMS.update(
        {
            "max_depth": 7,  # Range 6-8
            "learning_rate": 0.02,
            "n_estimators": 3000,  # More trees due to lower LR
            "early_stopping_rounds": 50,
        }
    )

    # =========================================================================
    # Caching Configuration
    # =========================================================================
    # Filenames for cached features
    CACHE_FILES = {
        "train_stream_a": os.path.join(WORKING_DIR, "features_train_streamA.parquet"),
        "train_stream_b": os.path.join(WORKING_DIR, "features_train_streamB.parquet"),
        "val_stream_a": os.path.join(WORKING_DIR, "features_val_streamA.parquet"),
        "val_stream_b": os.path.join(WORKING_DIR, "features_val_streamB.parquet"),
        "test_stream_a": os.path.join(WORKING_DIR, "features_test_streamA.parquet"),
        "test_stream_b": os.path.join(WORKING_DIR, "features_test_streamB.parquet"),
        # Numpy arrays for IDs and Labels corresponding to the parquets
        "train_stream_a_ids": os.path.join(
            WORKING_DIR, "features_train_streamA_ids.npy"
        ),
        "train_stream_a_y": os.path.join(WORKING_DIR, "features_train_streamA_y.npy"),
        "train_stream_b_ids": os.path.join(
            WORKING_DIR, "features_train_streamB_ids.npy"
        ),
        "train_stream_b_y": os.path.join(WORKING_DIR, "features_train_streamB_y.npy"),
        "val_stream_a_ids": os.path.join(WORKING_DIR, "features_val_streamA_ids.npy"),
        "val_stream_a_y": os.path.join(WORKING_DIR, "features_val_streamA_y.npy"),
        "val_stream_b_ids": os.path.join(WORKING_DIR, "features_val_streamB_ids.npy"),
        "val_stream_b_y": os.path.join(WORKING_DIR, "features_val_streamB_y.npy"),
        "test_stream_a_ids": os.path.join(WORKING_DIR, "features_test_streamA_ids.npy"),
        "test_stream_b_ids": os.path.join(WORKING_DIR, "features_test_streamB_ids.npy"),
    }
