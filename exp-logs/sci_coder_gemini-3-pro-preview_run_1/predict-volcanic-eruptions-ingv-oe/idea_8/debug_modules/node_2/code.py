import os
import pandas as pd
import numpy as np
import warnings
import shutil
from library import (
    config,
    utils,
    feature_engineering,
    vision_module,
    tabular_module,
    meta_learner,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Pipeline Demonstration ===")

    # ---------------------------------------------------------
    # 1. Configuration Overrides for Speed
    # ---------------------------------------------------------
    print("\n[1] Overriding configuration for fast demonstration...")
    # Reduce training iterations for immediate feedback
    config.CNN_CONFIG["epochs"] = 1
    config.CNN_CONFIG["batch_size"] = 4
    config.LGBM_PARAMS["n_estimators"] = 10
    config.LGBM_PARAMS["early_stopping_round"] = 5
    config.LGBM_PARAMS["num_leaves"] = 7
    config.LGBM_PARAMS["verbose"] = -1

    # Use a specific demo directory within working
    demo_dir = os.path.join(config.WORKING_DIR, "demo_execution")
    os.makedirs(demo_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 2. Create Mini Datasets (Subset of Metadata)
    # ---------------------------------------------------------
    print("\n[2] Creating mini datasets from metadata...")
    # Load full metadata
    df_train_full = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_test_full = pd.read_csv(config.TEST_METADATA_PATH)

    # Select a small subset: 10 training samples, 4 test samples
    # We need enough for a train/val split (e.g., 8 train, 2 val)
    df_mini_train = df_train_full.head(10).copy()
    df_mini_test = df_test_full.head(4).copy()

    # Save mini metadata
    mini_train_meta_path = os.path.join(demo_dir, "mini_train.csv")
    mini_test_meta_path = os.path.join(demo_dir, "mini_test.csv")

    df_mini_train.to_csv(mini_train_meta_path, index=False)
    df_mini_test.to_csv(mini_test_meta_path, index=False)

    print(f"    Mini Train: {len(df_mini_train)} samples")
    print(f"    Mini Test:  {len(df_mini_test)} samples")

    # ---------------------------------------------------------
    # 3. Feature Engineering & Data Loading
    # ---------------------------------------------------------
    print("\n[3] Running feature engineering on mini datasets...")

    # Define paths for demo features to avoid overwriting main cache
    train_tab_path = os.path.join(demo_dir, "train_features.parquet")
    train_spec_path = os.path.join(demo_dir, "train_specs.npy")
    test_tab_path = os.path.join(demo_dir, "test_features.parquet")
    test_spec_path = os.path.join(demo_dir, "test_specs.npy")

    # Process Training Data
    # load_cached_data=False forces re-processing for this demo
    df_train_feats, train_specs, train_targets = (
        feature_engineering.load_and_process_data(
            mini_train_meta_path,
            train_tab_path,
            train_spec_path,
            load_cached_data=False,
        )
    )

    # Process Test Data
    df_test_feats, test_specs, test_targets = feature_engineering.load_and_process_data(
        mini_test_meta_path, test_tab_path, test_spec_path, load_cached_data=False
    )

    # Validation of Feature Engineering
    assert len(df_train_feats) == 10, "Train feature count mismatch"
    assert train_specs.shape == (
        10,
        10,
        128,
        128,
    ), f"Train spectrogram shape mismatch: {train_specs.shape}"
    assert len(train_targets) == 10, "Train target count mismatch"
    assert len(df_test_feats) == 4, "Test feature count mismatch"
    assert test_specs.shape == (4, 10, 128, 128), "Test spectrogram shape mismatch"
    print("    Feature engineering validation passed.")

    # ---------------------------------------------------------
    # 4. Train/Validation Split
    # ---------------------------------------------------------
    print("\n[4] Splitting data into Train/Validation...")
    split_idx = 8  # 8 for training, 2 for validation

    X_train = df_train_feats.iloc[:split_idx]
    y_train = train_targets[:split_idx]
    X_val = df_train_feats.iloc[split_idx:]
    y_val = train_targets[split_idx:]

    specs_train = train_specs[:split_idx]
    specs_val = train_specs[split_idx:]

    print(f"    Train size: {len(X_train)}, Val size: {len(X_val)}")

    # ---------------------------------------------------------
    # 5. Tabular Model Training (LightGBM)
    # ---------------------------------------------------------
    print("\n[5] Training Tabular Model (LightGBM)...")
    fold_idx = 999  # Use a dummy fold index for demo

    lgb_model, tab_val_preds, tab_test_preds = tabular_module.train_tabular_model(
        X_train, y_train, X_val, y_val, X_test=df_test_feats, fold_idx=fold_idx
    )

    # Assertions
    assert len(tab_val_preds) == len(X_val)
    assert len(tab_test_preds) == len(df_test_feats)
    assert np.all(tab_val_preds >= 0), "Found negative predictions in Tabular Val"

    # Verify model loading works
    loaded_preds = tabular_module.predict_tabular_model(df_test_feats, fold_idx)
    assert np.allclose(
        tab_test_preds, loaded_preds
    ), "Loaded model predictions do not match returned predictions"
    print("    Tabular model training and verification passed.")

    # ---------------------------------------------------------
    # 6. Vision Model Training (CNN)
    # ---------------------------------------------------------
    print("\n[6] Training Vision Model (EfficientNet)...")

    vis_val_preds = vision_module.train_vision_model(
        specs_train, y_train, specs_val, y_val, fold_idx=fold_idx
    )

    # Assertions
    assert len(vis_val_preds) == len(X_val)
    assert np.all(vis_val_preds >= 0), "Found negative predictions in Vision Val"

    # Generate Test Predictions
    vis_test_preds = vision_module.predict_vision_model(test_specs, fold_idx)
    assert len(vis_test_preds) == len(df_test_feats)
    print("    Vision model training and verification passed.")

    # ---------------------------------------------------------
    # 7. Meta-Learner Training (Stacking)
    # ---------------------------------------------------------
    print("\n[7] Training Meta-Learner (Ridge Regression)...")

    # Train on OOF predictions
    meta_model, meta_score = meta_learner.train_meta_learner(
        tab_val_preds, vis_val_preds, y_val
    )

    # Predict on Test Set
    final_preds = meta_learner.predict_meta_learner(tab_test_preds, vis_test_preds)

    # Assertions
    assert len(final_preds) == len(df_test_feats)
    assert np.all(final_preds >= 0), "Found negative predictions in Meta-Learner"
    print(f"    Meta-Learner trained. Final Test Predictions: {final_preds}")

    # ---------------------------------------------------------
    # 8. Submission Generation
    # ---------------------------------------------------------
    print("\n[8] Generating Submission File...")
    sub_path = os.path.join(demo_dir, "submission.csv")

    utils.save_submission(df_test_feats["segment_id"].values, final_preds, sub_path)

    if os.path.exists(sub_path):
        print(f"    Submission saved successfully to {sub_path}")
        # Verify file content
        df_sub = pd.read_csv(sub_path)
        assert df_sub.shape == (4, 2), "Submission file shape mismatch"
        assert list(df_sub.columns) == [
            "segment_id",
            "time_to_eruption",
        ], "Submission columns mismatch"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
