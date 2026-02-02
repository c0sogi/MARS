import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

# Import from the provided library
import library.config
import library.features
import library.data
import library.model

# Set random seed for reproducibility
np.random.seed(42)


def main():
    print("--- Starting Demonstration Script ---")

    # 1. Optimize for Speed: Adjust Hyperparameters
    # We modify the configuration in memory to reduce training time for this demo.
    # The default is 3000 estimators, which is too slow for a quick check.
    print("Adjusting XGBoost parameters for speed...")
    library.config.XGB_PARAMS["n_estimators"] = 50
    library.config.XGB_PARAMS["learning_rate"] = (
        0.1  # Increase LR to compensate for fewer trees
    )
    print(f"XGB_PARAMS: {library.config.XGB_PARAMS}")

    # 2. Load Data
    # This uses the metadata CSVs to load features.
    # It calculates features from geometry files if cache is not found.
    # We set load_cached_data=True to use existing parquet files if available.
    print("\n--- Loading Datasets ---")
    df_train, df_val, df_test = library.data.load_datasets(load_cached_data=True)

    # Validate loaded data
    print(f"Train shape: {df_train.shape}")
    print(f"Val shape:   {df_val.shape}")
    print(f"Test shape:  {df_test.shape}")

    assert len(df_train) > 0, "Training data is empty."
    assert len(df_val) > 0, "Validation data is empty."
    assert len(df_test) > 0, "Test data is empty."

    # 3. Preprocess Targets
    # The metric is RMSLE, so we train on log1p(target).
    print("\n--- Preprocessing Targets (Log Transform) ---")
    df_train_log = library.data.preprocess_targets(df_train)
    df_val_log = library.data.preprocess_targets(df_val)

    # Verify transformation
    for target in library.config.TARGET_COLS:
        orig_val = df_train.iloc[0][target]
        trans_val = df_train_log.iloc[0][target]
        expected = np.log1p(orig_val)
        assert np.isclose(trans_val, expected), f"Log transform failed for {target}"

    # 4. Prepare Feature Matrices
    # This aligns columns and fills missing values.
    print("\n--- Preparing Matrices ---")
    X_train, y_train, X_val, y_val, X_test, feature_names = (
        library.data.prepare_matrices(df_train_log, df_val_log, df_test)
    )

    print(f"Number of selected features: {len(feature_names)}")
    assert (
        X_train.shape[1] == X_test.shape[1]
    ), "Train and Test feature counts mismatch."

    # 5. Train Models and Predict
    submission_data = {"id": df_test["id"].values}

    for target_name in library.config.TARGET_COLS:
        print(f"\nProcessing Target: {target_name}")

        # Select specific target series
        y_train_target = y_train[target_name]
        y_val_target = y_val[target_name]

        # Train
        model = library.model.train_target_model(
            X_train,
            y_train_target,
            X_val,
            y_val_target,
            target_name,
            early_stopping_rounds=10,
        )

        # Predict on Test Set (Log Space)
        preds_log = library.model.make_predictions(model, X_test)

        # Inverse Transform (Expm1) to get original scale
        preds_orig = library.data.inverse_transform_targets(preds_log)

        # Store results
        submission_data[target_name] = preds_orig

        # Basic sanity check on predictions
        assert not np.isnan(
            preds_orig
        ).any(), f"NaN predictions found for {target_name}"
        assert (
            preds_orig >= 0
        ).all(), f"Negative predictions found for {target_name} (physical quantities must be non-negative)"

    # 6. Generate Submission File
    print("\n--- Generating Submission File ---")
    submission_df = pd.DataFrame(submission_data)

    # Ensure column order matches requirements: id, formation_energy_ev_natom, bandgap_energy_ev
    cols_order = ["id"] + library.config.TARGET_COLS
    submission_df = submission_df[cols_order]

    output_path = os.path.join("working", "demo_submission.csv")
    submission_df.to_csv(output_path, index=False)

    print(f"Submission saved to: {output_path}")
    print("Head of submission:")
    print(submission_df.head())

    # Final Verification
    assert os.path.exists(output_path), "Output file was not created."
    loaded_sub = pd.read_csv(output_path)
    assert loaded_sub.shape == (len(df_test), 3), "Submission shape is incorrect."
    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    main()
