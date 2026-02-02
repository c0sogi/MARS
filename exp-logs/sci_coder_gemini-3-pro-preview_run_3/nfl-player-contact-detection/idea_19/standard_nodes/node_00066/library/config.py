import os

# Ensure the working directory for caching exists immediately
WORKING_DIR = "./working/idea_19"
os.makedirs(WORKING_DIR, exist_ok=True)


class Config:
    """
    Global configuration for the Physically-Disentangled Dual-Stream GBDT.
    """

    # Global Random Seed for Reproducibility
    SEED = 42

    # Directory Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = WORKING_DIR
    SUBMISSION_DIR = "./submission"

    # Ensure submission directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # File Path Configuration
    PATH_CONFIG = {
        # Raw Input Files
        "train_labels": os.path.join(INPUT_DIR, "train_labels.csv"),
        "train_tracking": os.path.join(INPUT_DIR, "train_player_tracking.csv"),
        "test_tracking": os.path.join(INPUT_DIR, "test_player_tracking.csv"),
        "train_helmets": os.path.join(INPUT_DIR, "train_baseline_helmets.csv"),
        "test_helmets": os.path.join(INPUT_DIR, "test_baseline_helmets.csv"),
        "train_video_meta": os.path.join(INPUT_DIR, "train_video_metadata.csv"),
        "test_video_meta": os.path.join(INPUT_DIR, "test_video_metadata.csv"),
        "sample_submission": os.path.join(INPUT_DIR, "sample_submission.csv"),
        # Generated Metadata (Splits)
        "metadata_train": os.path.join(METADATA_DIR, "train.csv"),
        "metadata_val": os.path.join(METADATA_DIR, "validation.csv"),
        "metadata_test": os.path.join(METADATA_DIR, "test.csv"),
        # Output
        "submission_path": os.path.join(SUBMISSION_DIR, "submission.csv"),
    }

    # Feature Engineering Configuration
    FEATURE_CONFIG = {
        # Exponential Temporal Pyramids for Tracking Data: t +/- [1, 2, 4, 8, 15]
        "tracking_lags": [1, 2, 4, 8, 15],
        # Sparse Temporal Pyramids for Visual Data: t +/- [4, 8, 15]
        "visual_lags": [4, 8, 15],
        # Columns to load from tracking data
        "tracking_cols": [
            "x_position",
            "y_position",
            "speed",
            "direction",
            "orientation",
            "acceleration",
            "sa",
            "jersey_number",
            "team",
        ],
        # Stream A: Interaction Model (Player-Player)
        # Uses Relational Scalars + Visual Pyramids. No Ego-centric projection.
        "stream_a": {
            "use_visuals": True,
            "use_ego_centric": False,
            "use_relational": True,
        },
        # Stream B: Impact Model (Player-Ground)
        # Uses Raw Field-Centric Anchor + Ego-Centric Augmentation. No Visuals.
        "stream_b": {
            "use_visuals": False,
            "use_ego_centric": True,
            "use_relational": False,
        },
    }

    # Training Configuration
    TRAIN_CONFIG = {
        # Targeted Majority Undersampling: 10 Negatives for every 1 Positive
        "neg_pos_ratio": 10,
        # Training Loop
        "num_boost_round": 5000,
        "early_stopping_rounds": 50,
        "verbose_eval": 100,
        # Validation
        "optimize_threshold": True,
    }

    # XGBoost Parameters - Stream A (Interaction)
    # Standard regularization for relational/visual features
    XGB_PARAMS_STREAM_A = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "gpu_hist",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_jobs": 12,
        "random_state": SEED,
    }

    # XGBoost Parameters - Stream B (Impact)
    # Deeper trees and relaxed regularization to capture complex kinematic signatures
    XGB_PARAMS_STREAM_B = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "gpu_hist",
        "learning_rate": 0.05,
        "max_depth": 10,  # Deeper trees for raw/ego-centric interactions
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.01,  # Relaxed
        "reg_lambda": 0.1,  # Relaxed
        "n_jobs": 12,
        "random_state": SEED,
    }
