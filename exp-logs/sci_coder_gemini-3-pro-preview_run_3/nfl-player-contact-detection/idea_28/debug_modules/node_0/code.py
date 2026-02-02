import os
import sys
import pandas as pd
import numpy as np
import shutil
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure working directory exists
os.makedirs("./working", exist_ok=True)

# ==============================================================================
# 1. Import Library Modules
# ==============================================================================
# We import config first to modify mutable configurations (dicts) before other modules use them.
import library.config as config
from library.physics_engine import (
    calculate_euclidean_distance,
    calculate_iou_metrics,
    project_ego_velocity,
)
from library.data_manager import DataManager
from library.feature_builder import FeatureBuilder
from library.model_trainer import ModelTrainer
from library.utils import seed_everything

# Set global seed
seed_everything(42)


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # ==============================================================================
    # 2. Physics Engine Verification
    # ==============================================================================
    print("\n[1/5] Verifying Physics Engine...")

    # Test Euclidean Distance
    x1, y1 = np.array([0, 0]), np.array([0, 0])
    x2, y2 = np.array([3, 6]), np.array([4, 8])  # 3-4-5 triangle, 6-8-10 triangle
    dists = calculate_euclidean_distance(x1, y1, x2, y2)
    assert np.allclose(dists, [5.0, 10.0]), f"Euclidean distance failed: {dists}"
    print("  - Euclidean distance: OK")

    # Test IoU (Intersection over Union)
    # Box: [left, width, top, height]
    box1 = np.array([0, 10, 0, 10])  # 10x10 square at (0,0)
    box2 = np.array(
        [5, 10, 0, 10]
    )  # 10x10 square at (5,0), Overlap width=5, height=10. Area=50.
    # Union = 100 + 100 - 50 = 150. IoU = 50/150 = 1/3
    iou = calculate_iou_metrics(box1, box2)
    assert np.isclose(iou, 1 / 3), f"IoU calculation failed: {iou}"
    print("  - IoU calculation: OK")

    # Test Ego Velocity Projection
    # Player moving North (Direction 0/360), Facing North (Orientation 0) -> Surge = Speed, Sway = 0
    # Note: Physics engine assumes standard trig or specific N/S convention.
    # Let's check the implementation: theta = direction - orientation.
    # If dir=0, orient=0, theta=0. cos(0)=1 (Surge), sin(0)=0 (Sway).
    speed = 10.0
    direction = 0.0
    orientation = 0.0
    v_surge, v_sway = project_ego_velocity(speed, direction, orientation)
    assert np.isclose(v_surge, 10.0) and np.isclose(
        v_sway, 0.0
    ), f"Ego projection failed: {v_surge}, {v_sway}"

    # Player moving East (90), Facing North (0) -> Surge = 0, Sway = 10 (Right)
    # theta = 90. cos(90)=0, sin(90)=1.
    v_surge, v_sway = project_ego_velocity(speed, 90.0, 0.0)
    assert np.isclose(v_surge, 0.0, atol=1e-5) and np.isclose(
        v_sway, 10.0, atol=1e-5
    ), "Ego projection (90 deg) failed"
    print("  - Ego velocity projection: OK")

    # ==============================================================================
    # 3. Data Subsetting (Optimization for Speed)
    # ==============================================================================
    print("\n[2/5] Preparing Subset Data for Speed...")

    # Load original metadata to sample from
    df_train_meta = pd.read_csv(config.METADATA_PATHS["train"])

    # Pick 2 unique game_plays for training, 1 for validation
    unique_plays = df_train_meta["game_play"].unique()
    if len(unique_plays) < 3:
        raise ValueError("Not enough unique plays in metadata for demo split.")

    train_plays = unique_plays[:2]
    val_plays = unique_plays[2:3]

    df_mini_train = df_train_meta[df_train_meta["game_play"].isin(train_plays)].copy()
    df_mini_val = df_train_meta[df_train_meta["game_play"].isin(val_plays)].copy()

    # Save mini metadata
    mini_train_path = "./working/mini_train.csv"
    mini_val_path = "./working/mini_validation.csv"
    df_mini_train.to_csv(mini_train_path, index=False)
    df_mini_val.to_csv(mini_val_path, index=False)

    print(
        f"  - Created mini train set: {len(df_mini_train)} rows ({len(train_plays)} plays)"
    )
    print(f"  - Created mini val set: {len(df_mini_val)} rows ({len(val_plays)} plays)")

    # MONKEY PATCH CONFIGURATION
    # We modify the dictionaries in the imported config module in-place.
    config.METADATA_PATHS["train"] = mini_train_path
    config.METADATA_PATHS["validation"] = mini_val_path

    # Also patch XGBoost params to run extremely fast
    config.XGB_PARAMS_STREAM_A["n_estimators"] = 5
    config.XGB_PARAMS_STREAM_A["max_depth"] = 3
    config.XGB_PARAMS_STREAM_B["n_estimators"] = 5
    config.XGB_PARAMS_STREAM_B["max_depth"] = 3

    # Patch working dir to avoid conflicts with real runs
    # Note: Classes use config.WORKING_DIR directly, so we patch it
    config.WORKING_DIR = "./working/demo_run"
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    print("  - Configuration patched for demo execution.")

    # ==============================================================================
    # 4. Data Manager Execution
    # ==============================================================================
    print("\n[3/5] Running DataManager...")

    # Initialize DataManagers
    dm_train = DataManager(mode="train", debug=True)
    dm_val = DataManager(mode="validation", debug=True)

    # Load and Process Data
    # Note: load_cached=False forces processing from scratch using our mini metadata
    df_train_a, df_train_b = dm_train.load_data(load_cached=False)
    df_val_a, df_val_b = dm_val.load_data(load_cached=False)

    # Verification
    assert not df_train_a.empty, "Stream A Training Data is empty"
    assert not df_train_b.empty, "Stream B Training Data is empty"
    assert "iou_sideline" in df_train_a.columns, "Stream A missing IoU features"
    assert (
        "v_surge" not in df_train_b.columns
    ), "Stream B should not have features yet (only merged)"

    print(f"  - Train Stream A shape: {df_train_a.shape}")
    print(f"  - Train Stream B shape: {df_train_b.shape}")

    # ==============================================================================
    # 5. Feature Builder Execution
    # ==============================================================================
    print("\n[4/5] Running FeatureBuilder...")

    fb_train = FeatureBuilder(mode="train")
    # We reuse the same builder class logic, but usually we instantiate per mode to manage cache keys
    # To keep it simple, we just instantiate one for validation with mode='validation'
    fb_val = FeatureBuilder(mode="validation")

    # Build Features
    data_train_a, data_train_b = fb_train.build_features(
        df_train_a, df_train_b, load_cached=False
    )
    data_val_a, data_val_b = fb_val.build_features(
        df_val_a, df_val_b, load_cached=False
    )

    # Verification
    assert (
        "distance" in data_train_a["X"].columns
    ), "Feature 'distance' missing in Stream A"
    assert (
        "v_surge" in data_train_b["X"].columns
    ), "Feature 'v_surge' missing in Stream B"
    assert data_train_a["X"].shape[0] == len(
        data_train_a["y"]
    ), "X and y length mismatch in Stream A"

    # Check for Lagged Features (Temporal Pyramids)
    lag_col = [c for c in data_train_a["X"].columns if "_lag1" in c]
    assert len(lag_col) > 0, "Temporal features (lags) not generated."

    print(f"  - Features built. Stream A Train X: {data_train_a['X'].shape}")

    # ==============================================================================
    # 6. Model Training & Inference
    # ==============================================================================
    print("\n[5/5] Running ModelTrainer...")

    trainer = ModelTrainer(debug=True)

    # Train
    models, thresholds = trainer.train_and_evaluate(
        data_train_a, data_val_a, data_train_b, data_val_b
    )

    assert models["A"] is not None, "Model A failed to train"
    assert models["B"] is not None, "Model B failed to train"
    print(f"  - Optimal Thresholds: A={thresholds['A']:.2f}, B={thresholds['B']:.2f}")

    # Generate Submission
    # For demo purposes, we treat the validation set as the test set
    # In a real scenario, we would load test metadata and process it similarly
    print("  - Generating submission using validation set as proxy for test...")

    # Patch submission dir
    config.SUBMISSION_DIR = "./working/demo_run/submission"
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    trainer.generate_submission(models, thresholds, data_val_a, data_val_b)

    sub_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(sub_path), "Submission file was not created."

    df_sub = pd.read_csv(sub_path)
    print(f"  - Submission created at {sub_path}")
    print(f"  - Submission shape: {df_sub.shape}")
    print(f"  - Sample rows:\n{df_sub.head(3)}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
