import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

# ==========================================
# Global Path & Environment Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_14"

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# Input File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output/Cache Paths
CACHE_DIR = WORKING_DIR
MODEL_OUTPUT_DIR = os.path.join(CACHE_DIR, "models")
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure output directories exist
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Global Constants
# ==========================================
SEED = 42
K_NEIGHBORS = 3
GATING_DIST = 2.5
WINDOW_OFFSET = 10  # +/- 10 steps (approx +/- 1.0 second context)

# ==========================================
# Feature Configuration
# ==========================================


@dataclass
class FeatureConfig:
    """
    Configuration for feature engineering, defining input columns and generated feature names.
    """

    # Columns to read from raw tracking data
    tracking_cols: List[str] = field(
        default_factory=lambda: [
            "game_play",
            "step",
            "nfl_player_id",
            "x_position",
            "y_position",
            "speed",
            "direction",
            "orientation",
            "acceleration",
            "sa",
        ]
    )

    # Base kinematic features for the interacting pair (or Player-Ground)
    # Jerk is derived from acceleration difference over time
    pair_features: List[str] = field(
        default_factory=lambda: [
            "distance",
            "speed_p1",
            "speed_p2",
            "speed_diff",
            "accel_p1",
            "accel_p2",
            "accel_diff",
            "jerk_p1",
            "jerk_p2",
            "orientation_diff",
            "direction_diff",
        ]
    )

    # Columns to apply temporal windowing to (history/future)
    window_target_cols: List[str] = field(
        default_factory=lambda: [
            "speed_p1",
            "speed_p2",
            "distance",
            "accel_p1",
            "accel_p2",
        ]
    )

    # Primitives for Ordered Neighbor features (Relative Kinematics)
    neighbor_primitives: List[str] = field(
        default_factory=lambda: ["dist", "speed_rel", "closing_speed", "accel_rel"]
    )

    @property
    def neighbor_features(self) -> List[str]:
        """
        Generate feature names for the K nearest neighbors.
        Format: n{k}_{primitive} (e.g., n1_dist, n1_speed_rel)
        """
        feats = []
        for k in range(1, K_NEIGHBORS + 1):
            for prim in self.neighbor_primitives:
                feats.append(f"n{k}_{prim}")
        return feats

    @property
    def window_features(self) -> List[str]:
        """
        Generate feature names for temporal windows.
        Format: {col}_t{step} (e.g., speed_p1_t-5)
        """
        feats = []
        # Generate steps from -WINDOW_OFFSET to +WINDOW_OFFSET, skipping 0 (current step is in pair_features)
        steps = [i for i in range(-WINDOW_OFFSET, WINDOW_OFFSET + 1) if i != 0]
        for col in self.window_target_cols:
            for t in steps:
                feats.append(f"{col}_t{t:+d}")
        return feats

    @property
    def all_features(self) -> List[str]:
        """
        Returns the complete list of features to be used by the model.
        Includes Pair Kinematics, Temporal Windows, Ordered Neighbors, and Ground Indicator.
        """
        return (
            self.pair_features
            + self.window_features
            + self.neighbor_features
            + ["is_ground"]
        )


# ==========================================
# Model Configuration
# ==========================================


@dataclass
class ModelConfig:
    """
    Configuration for the Heterogeneous Ensemble (LightGBM + XGBoost).
    """

    # LightGBM Params (Leaf-wise growth)
    lgbm_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.02,  # Lower LR for stability with deep trees
            "num_leaves": 256,  # High capacity for complex interactions
            "max_depth": 10,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "n_estimators": 3000,
            "verbose": -1,
            "random_state": SEED,
            "is_unbalance": True,  # Handle class imbalance automatically
            "n_jobs": 12,
        }
    )

    # XGBoost Params (Level-wise growth)
    xgb_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "learning_rate": 0.02,
            "max_depth": 10,  # Deep trees
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "n_estimators": 3000,
            "random_state": SEED,
            "scale_pos_weight": 10,  # Moderate balancing, refined by hard negative mining
            "tree_method": "hist",
            "device": "cuda",  # Use GPU
            "n_jobs": 12,
        }
    )

    # Training Loop Settings
    early_stopping_rounds: int = 100
    verbose_eval: int = 100

    # Mining Strategy Settings
    scout_n_estimators: int = 500  # Faster, lighter model for initial mining pass
    hard_negative_threshold: float = 0.05  # Threshold to identify hard negatives
