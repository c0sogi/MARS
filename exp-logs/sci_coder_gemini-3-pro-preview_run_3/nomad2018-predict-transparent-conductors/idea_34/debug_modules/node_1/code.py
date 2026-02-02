import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Import provided library components
from library.config import (
    TRAIN_METADATA_PATH,
    TEST_METADATA_PATH,
    TARGET_COLS,
    SUBMISSION_PATH,
    RANDOM_SEED,
)
from library.features import generate_features, process_structure
from library.model import DualTargetRegressor
from library.data_loader import inverse_transform_targets

# Configuration for the demonstration
DEMO_SAMPLE_SIZE = 50  # Small subset for speed
TEST_SAMPLE_SIZE = 10
FAST_MODEL_PARAMS = {
    "n_estimators": 10,
    "learning_rate": 0.1,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": 1,  # Avoid heavy parallelism in demo
    "random_state": RANDOM_SEED,
    "objective": "reg:squarederror",
    "tree_method": "hist",
}


def run_demo_pipeline():
    print(">>> Starting End-to-End Demonstration Pipeline")

    # ---------------------------------------------------------
    # 1. Load and Sample Data
    # ---------------------------------------------------------
    print(f"\n[1/6] Loading metadata from {TRAIN_METADATA_PATH}...")
    if not os.path.exists(TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Train metadata not found at {TRAIN_METADATA_PATH}")

    df_full = pd.read_csv(TRAIN_METADATA_PATH)

    # Sample a subset for quick demonstration
    df_demo = df_full.sample(n=DEMO_SAMPLE_SIZE, random_state=RANDOM_SEED).copy()
    print(f"      Subsampled {len(df_demo)} samples from {len(df_full)} total.")

    # ---------------------------------------------------------
    # 2. Verify Feature Extraction Logic
    # ---------------------------------------------------------
    print("\n[2/6] Verifying atomic structure processing...")
    # Test on the first sample to ensure ASE reading and feature logic works
    sample_row = df_demo.iloc[0]
    print(f"      Processing sample file: {sample_row['file_path']}")

    single_feat = process_structure(sample_row)

    if single_feat is None:
        print(
            "      WARNING: Failed to read structure. This may be due to file format issues."
        )
        # We continue, as generate_features handles failures gracefully (returns empty dicts)
    else:
        print(f"      Success! Extracted {len(single_feat)} features.")
        # Basic validation of expected keys
        expected_keys = ["volume_per_atom", "density"]
        for k in expected_keys:
            assert k in single_feat, f"Missing expected feature key: {k}"

    # ---------------------------------------------------------
    # 3. Generate Features for Training Set
    # ---------------------------------------------------------
    print("\n[3/6] Generating features for demo training set...")
    # We use a custom split_name to avoid overwriting the main cache files
    df_geo_features = generate_features(
        df_demo,
        split_name="demo_train",
        load_cached_data=False,  # Force re-computation for demo
    )

    # Merge tabular metadata with geometric features
    # Exclude ID, file_path, and targets from X
    exclude_cols = ["id", "file_path"] + TARGET_COLS
    tabular_cols = [c for c in df_demo.columns if c not in exclude_cols]

    X = pd.concat(
        [
            df_demo[tabular_cols].reset_index(drop=True),
            df_geo_features.reset_index(drop=True),
        ],
        axis=1,
    )

    # Prepare Targets (Log Transform)
    y = np.log1p(df_demo[TARGET_COLS].reset_index(drop=True))

    # Split into Train/Validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"      Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # ---------------------------------------------------------
    # 4. Train Model
    # ---------------------------------------------------------
    print("\n[4/6] Training DualTargetRegressor (XGBoost)...")
    model = DualTargetRegressor(params=FAST_MODEL_PARAMS)

    # Fit the model
    model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        early_stopping_rounds=10,
        verbose=True,
    )

    # ---------------------------------------------------------
    # 5. Evaluate Model
    # ---------------------------------------------------------
    print("\n[5/6] Evaluating on Validation Set...")
    val_preds = model.predict(X_val)  # Returns predictions in original scale (eV)

    # Inverse transform ground truth for comparison
    y_val_orig = inverse_transform_targets(y_val)

    for target in TARGET_COLS:
        rmse = np.sqrt(mean_squared_error(y_val_orig[target], val_preds[target]))
        print(f"      Target: {target:<30} | RMSE: {rmse:.4f} eV")

        # Validation check: RMSE should be a finite positive number
        assert np.isfinite(rmse), f"RMSE for {target} is not finite"
        assert rmse >= 0, f"RMSE for {target} is negative"

    # ---------------------------------------------------------
    # 6. Generate Submission (Test Set)
    # ---------------------------------------------------------
    print("\n[6/6] Generating Submission for Test Subset...")
    if not os.path.exists(TEST_METADATA_PATH):
        raise FileNotFoundError(f"Test metadata not found at {TEST_METADATA_PATH}")

    df_test_full = pd.read_csv(TEST_METADATA_PATH)
    df_test_demo = df_test_full.sample(
        n=TEST_SAMPLE_SIZE, random_state=RANDOM_STATE
    ).copy()

    # Generate features for test subset
    df_test_geo = generate_features(
        df_test_demo, split_name="demo_test", load_cached_data=False
    )

    X_test = pd.concat(
        [
            df_test_demo[tabular_cols].reset_index(drop=True),
            df_test_geo.reset_index(drop=True),
        ],
        axis=1,
    )

    # Predict
    test_preds = model.predict(X_test)

    # Attach IDs
    submission = pd.DataFrame()
    submission["id"] = df_test_demo["id"].values
    submission[TARGET_COLS[0]] = test_preds[TARGET_COLS[0]].values
    submission[TARGET_COLS[1]] = test_preds[TARGET_COLS[1]].values

    print("\n      Generated Submission Sample:")
    print(submission.head())

    # Save (Demonstration only, usually we save full submission)
    demo_sub_path = "working/demo_submission.csv"
    submission.to_csv(demo_sub_path, index=False)
    print(f"      Saved demo submission to {demo_sub_path}")

    print("\n>>> Demonstration Pipeline Completed Successfully.")


if __name__ == "__main__":
    run_demo_pipeline()
