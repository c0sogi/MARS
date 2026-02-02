import os
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import XGB_PARAMS, SUBMISSION_PATH, WORKING_DIR
from library.data_manager import process_dataset
from library.preprocessor import preprocess_features, TargetTransformer
from library.regressor import DualTargetRegressor, generate_submission


def main():
    print("=== Starting Library Usage Demonstration ===")

    # 1. Data Management
    # We use debug_limit=20 to process only the first 20 samples for speed.
    # load_cached_data=False forces the feature generation logic to run.
    print("\n[Step 1] Processing Dataset (Feature Generation)...")
    X_train, y_train, X_val, y_val, X_test, test_ids = process_dataset(
        load_cached_data=False, debug_limit=20
    )

    # Verification of Data Loading
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape}")

    assert len(X_train) > 0, "X_train should not be empty"
    assert len(X_train) == len(y_train), "X_train and y_train row counts must match"
    assert X_train.shape[1] > 0, "Features should be generated"
    # Since we requested 20 samples, and the split is roughly 80/20,
    # train should have ~16 and val ~4. Test has its own limit.
    # Note: process_dataset applies debug_limit to train, val, and test metadata separately.
    assert len(X_train) <= 20, "Debug limit should constrain train size"
    assert len(X_test) <= 20, "Debug limit should constrain test size"

    # 2. Preprocessing
    print("\n[Step 2] Preprocessing Features...")
    # Define a separate cache dir for this demo to avoid conflicts
    demo_cache_dir = os.path.join(WORKING_DIR, "demo_execution")

    X_train_clean, X_val_clean, X_test_clean, cleaner = preprocess_features(
        X_train, X_val, X_test, load_cached_data=False, cache_dir=demo_cache_dir
    )

    # Verification of Preprocessing
    assert (
        X_train_clean.shape[0] == X_train.shape[0]
    ), "Row count shouldn't change during cleaning"
    assert (
        X_train_clean.shape[1] <= X_train.shape[1]
    ), "Feature count should decrease or stay same"
    assert os.path.exists(
        os.path.join(demo_cache_dir, "preprocessor_state.json")
    ), "Preprocessor state should be saved"

    # Verify TargetTransformer logic independently
    print("Verifying TargetTransformer...")
    tt = TargetTransformer()
    dummy_y = np.array([0.0, 1.0, 10.0])
    trans_y = tt.transform(dummy_y)
    inv_y = tt.inverse_transform(trans_y)
    assert np.allclose(dummy_y, inv_y), "TargetTransformer inverse transform failed"
    assert np.allclose(
        trans_y, np.log1p(dummy_y)
    ), "TargetTransformer transform logic incorrect"

    # 3. Model Training
    print("\n[Step 3] Training Regressor...")
    # Modify params for speed in this demo
    demo_params = XGB_PARAMS.copy()
    demo_params["n_estimators"] = 10  # Reduced for speed
    demo_params["max_depth"] = 3

    regressor = DualTargetRegressor(params=demo_params)

    regressor.fit(
        X_train_clean,
        y_train,
        X_val_clean,
        y_val,
        early_stopping_rounds=5,
        verbose=True,
    )

    # 4. Evaluation
    print("\n[Step 4] Evaluating Model...")
    metrics = regressor.evaluate(X_val_clean, y_val)

    # Verification of Metrics
    assert "formation_energy_ev_natom" in metrics, "Missing formation energy metric"
    assert "bandgap_energy_ev" in metrics, "Missing bandgap energy metric"
    assert metrics["formation_energy_ev_natom"] >= 0, "RMSLE cannot be negative"

    # 5. Submission Generation
    print("\n[Step 5] Generating Submission...")
    # Use a demo path to avoid overwriting main submission if needed,
    # though the requirement says use provided library functions.
    # We will use the default path but verify it exists.
    demo_submission_path = os.path.join(WORKING_DIR, "demo_submission.csv")

    generate_submission(
        regressor, X_test_clean, test_ids, output_path=demo_submission_path
    )

    # Verification of Submission
    assert os.path.exists(demo_submission_path), "Submission file was not created"

    sub_df = pd.read_csv(demo_submission_path)
    assert "id" in sub_df.columns, "Submission missing 'id' column"
    assert "formation_energy_ev_natom" in sub_df.columns, "Submission missing target 1"
    assert "bandgap_energy_ev" in sub_df.columns, "Submission missing target 2"
    assert len(sub_df) == len(test_ids), "Submission row count mismatch"

    # Check for non-negative predictions (physically required)
    assert (
        sub_df["formation_energy_ev_natom"] >= 0
    ).all(), "Negative formation energy predicted"
    assert (sub_df["bandgap_energy_ev"] >= 0).all(), "Negative bandgap energy predicted"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
