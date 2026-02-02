import os
import pandas as pd
import numpy as np
from library.config import Config
from library.train import train_model


def run_demo():
    print("=== Starting Demonstration of Hybrid GNN-GBDT Pipeline ===")

    # --- 1. Optimize Configuration for Speed ---
    # We modify the Config class attributes directly to run a fast demo.

    # Process only a small number of samples to ensure the GNN feature extraction
    # and training complete quickly.
    Config.DEBUG_SAMPLE_SIZE = 20

    # Reduce XGBoost estimators for rapid training demonstration.
    Config.XGB_PARAMS["n_estimators"] = 10
    Config.XGB_PARAMS["n_jobs"] = 1  # Avoid overhead for small data

    # Change cache paths to avoid overwriting or loading real training artifacts.
    # This ensures we actually run the feature extraction logic in this demo.
    Config.TRAIN_FEATURES_CACHE = os.path.join(
        Config.WORKING_DIR, "demo_train_features.parquet"
    )
    Config.VAL_FEATURES_CACHE = os.path.join(
        Config.WORKING_DIR, "demo_val_features.parquet"
    )
    Config.TEST_FEATURES_CACHE = os.path.join(
        Config.WORKING_DIR, "demo_test_features.parquet"
    )

    print(f"Debug Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"XGBoost Estimators: {Config.XGB_PARAMS['n_estimators']}")
    print(f"Temporary Cache Path: {Config.TRAIN_FEATURES_CACHE}")

    # --- 2. Execute Training Pipeline ---
    # We set load_cached_data=False to force the execution of the GNN feature extractor
    # on our small subset, verifying that component works.
    print("\n--- Executing train_model() ---")
    model, metrics = train_model(load_cached_data=False)

    # --- 3. Verify Outputs ---
    print("\n--- Verifying Results ---")

    # A. Verify Model Training
    assert model is not None, "Model object should not be None."
    # Check if models for all targets exist
    for target in Config.TARGET_COLS:
        assert target in model.models, f"Model for target '{target}' was not trained."
        print(f"[Pass] Model for {target} exists.")

    # B. Verify Metrics
    assert isinstance(metrics, dict), "Metrics should be a dictionary."
    assert "Mean_RMSLE" in metrics, "Mean_RMSLE missing from metrics."
    print(
        f"[Pass] Evaluation Metrics computed. Mean RMSLE: {metrics['Mean_RMSLE']:.4f}"
    )

    # C. Verify Submission File
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    df_sub = pd.read_csv(submission_path)

    # Check dimensions: Should have rows equal to test set size (or debug size if applied to test)
    # Note: prepare_feature_matrix applies DEBUG_SAMPLE_SIZE to test data as well.
    expected_rows = Config.DEBUG_SAMPLE_SIZE
    assert (
        len(df_sub) == expected_rows
    ), f"Submission has {len(df_sub)} rows, expected {expected_rows}."

    # Check columns
    expected_cols = ["id"] + Config.TARGET_COLS
    for col in expected_cols:
        assert col in df_sub.columns, f"Column {col} missing from submission."

    # Check for valid values (no NaNs, positive energies)
    assert not df_sub.isnull().values.any(), "Submission contains NaN values."
    # Energies are usually positive, especially bandgap
    if "bandgap_energy_ev" in df_sub.columns:
        assert (
            df_sub["bandgap_energy_ev"] >= 0
        ).all(), "Negative bandgap energies predicted."

    print(f"[Pass] Submission file structure verified: {df_sub.shape}")
    print("\nSample Predictions:")
    print(df_sub.head())

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
