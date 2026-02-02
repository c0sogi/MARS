import os
import pandas as pd
import numpy as np
import shutil
import lightgbm as lgb

# Import provided library modules
import library.config
import library.utils
import library.data_loader
import library.model_trainer


def main():
    print("Starting Volcano Eruption Prediction Pipeline Demo...")

    # 1. Setup and Reproducibility
    library.utils.seed_everything(42)

    # 2. Patch Configuration for Speed
    # The default N_ESTIMATORS is 6000, which is too slow for a demo.
    # We patch the module-level variables in model_trainer where they are imported.
    print("Patching configuration for fast execution...")
    library.model_trainer.N_ESTIMATORS = 10
    library.model_trainer.EARLY_STOPPING_ROUNDS = 5

    # Update the parameter dictionary to ensure silence and consistency
    library.config.LGBM_PARAMS.update({"verbosity": -1, "n_estimators": 10})

    # 3. Data Loading & Feature Engineering Demo
    print("\n--- Step 1: Data Loading & Feature Engineering ---")

    # Load a small subset of training data (20 samples)
    # load_cached_data=False ensures we trigger the feature engineering logic
    print("Loading training data (subset)...")
    X_train, y_train = library.data_loader.load_dataset(
        "train", load_cached_data=False, debug_n=20
    )

    # Load a small subset of test data (10 samples)
    print("Loading test data (subset)...")
    X_test, y_test = library.data_loader.load_dataset(
        "test", load_cached_data=False, debug_n=10
    )

    # Validation of Data Loading
    assert isinstance(X_train, pd.DataFrame), "X_train should be a DataFrame"
    assert isinstance(y_train, pd.Series), "y_train should be a Series"
    assert len(X_train) == 20, f"Expected 20 training samples, got {len(X_train)}"
    assert len(X_test) == 10, f"Expected 10 test samples, got {len(X_test)}"
    assert y_test is None, "y_test should be None for test set"
    assert X_train.index.name == "segment_id", "Index should be segment_id"

    # Check for specific features to ensure engineering worked (e.g., View A, B, C features)
    cols = X_train.columns
    assert "sensor_1_raw_mean" in cols, "Missing View C feature (Raw stats)"
    assert "sensor_1_trend_mean" in cols, "Missing View A feature (Trend stats)"
    assert "sensor_1_txt_std" in cols, "Missing View B feature (Texture stats)"

    print("Data loaded and validated successfully.")

    # 4. Model Training Demo (Single Fold)
    print("\n--- Step 2: Single Model Training ---")

    # Manual split for demonstration
    split_idx = 15
    X_tr_fold = X_train.iloc[:split_idx]
    y_tr_fold = y_train.iloc[:split_idx]
    X_val_fold = X_train.iloc[split_idx:]
    y_val_fold = y_train.iloc[split_idx:]

    print(f"Training LightGBM on {len(X_tr_fold)} samples...")
    model = library.model_trainer.train_lgbm_fold(
        X_tr_fold, y_tr_fold, X_val_fold, y_val_fold
    )

    assert isinstance(model, lgb.Booster), "Model should be a LightGBM Booster"

    # Test prediction
    preds = model.predict(X_val_fold)
    assert len(preds) == len(X_val_fold), "Prediction length mismatch"
    print("Single model training and prediction successful.")

    # 5. Cross-Validation Pipeline Demo
    print("\n--- Step 3: Stratified Cross-Validation ---")

    # StratifiedKFold requires enough samples per bin.
    # With only 20 samples and 15 bins (hardcoded in model_trainer), it will fail.
    # We replicate the data to create a synthetic dataset of 100 samples for the mechanism test.
    print("Creating synthetic dataset for CV mechanism test...")
    X_cv = pd.concat([X_train] * 5)
    y_cv = pd.concat([y_train] * 5)

    # Reset index to ensure uniqueness for OOF assignment
    # We assign new dummy segment_ids
    X_cv.index = range(1, len(X_cv) + 1)
    X_cv.index.name = "segment_id"
    y_cv.index = X_cv.index

    print(f"Running CV on {len(X_cv)} samples...")
    oof_preds, test_preds_avg, models = library.model_trainer.run_stratified_cv(
        X_cv, y_cv, n_folds=5, test_X=X_test
    )

    # Validation of CV results
    assert len(oof_preds) == len(X_cv), "OOF predictions length mismatch"
    assert len(models) == 5, "Should have trained 5 models"
    assert test_preds_avg is not None, "Test predictions should not be None"
    assert len(test_preds_avg) == len(X_test), "Test predictions length mismatch"
    print("Cross-Validation pipeline executed successfully.")

    # 6. Submission Generation Demo
    print("\n--- Step 4: Submission Generation ---")

    test_ids = X_test.index
    library.model_trainer.generate_submission(test_ids, test_preds_avg)

    assert os.path.exists(library.config.SUBMISSION_PATH), "Submission file not found"

    # Verify content
    sub_df = pd.read_csv(library.config.SUBMISSION_PATH)
    assert list(sub_df.columns) == [
        "segment_id",
        "time_to_eruption",
    ], "Incorrect submission columns"
    assert len(sub_df) == len(X_test), "Submission row count mismatch"
    print(f"Submission generated at {library.config.SUBMISSION_PATH}")

    # 7. Utils Demo (Artifacts)
    print("\n--- Step 5: Utils (Artifact Management) ---")

    dummy_data = {"a": [1, 2], "b": [3, 4]}
    dummy_df = pd.DataFrame(dummy_data)
    artifact_path = os.path.join(library.config.WORKING_DIR, "demo_artifact.parquet")

    library.utils.save_artifact(dummy_df, artifact_path)
    assert os.path.exists(artifact_path), "Artifact not saved"

    loaded_df = library.utils.load_artifact(artifact_path)
    pd.testing.assert_frame_equal(dummy_df, loaded_df)
    print("Artifact save/load verified.")

    print("\nAll demonstrations completed successfully!")


if __name__ == "__main__":
    main()
