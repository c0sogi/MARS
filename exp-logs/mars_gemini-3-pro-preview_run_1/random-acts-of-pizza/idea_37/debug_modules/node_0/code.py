import os
import shutil
import pandas as pd
import numpy as np
import torch

# Import library modules
import library.config as config
import library.utils as utils
import library.model_rf as model_rf
import library.model_mlp as model_mlp


def main():
    print("Starting demonstration script...")

    # 1. Configuration Overrides for Speed
    # We modify the configuration dictionaries in-place to ensure fast execution for this demo.
    print("Overriding configuration for fast demonstration...")

    # Random Forest overrides: Reduce estimators and parallelize
    config.RF_PARAMS["n_estimators"] = 10
    config.RF_PARAMS["n_jobs"] = 4

    # MLP overrides: Reduce training duration and model size
    config.MLP_PARAMS["epochs"] = 2
    config.MLP_PARAMS["hidden_dims"] = [64, 32]  # Smaller capacity for demo
    config.MLP_PARAMS["batch_size"] = 64
    config.MLP_PARAMS["patience"] = 1  # Aggressive early stopping

    # 2. Clear Cache to demonstrate Feature Engineering
    # The DataBuilder defaults to loading from cache if files exist.
    # To demonstrate the feature generation logic (SBERT, TF-IDF, etc.), we clear the cache.
    if os.path.exists(config.CACHE_DIR):
        print(f"Clearing cache directory: {config.CACHE_DIR}")
        shutil.rmtree(config.CACHE_DIR)
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # 3. Random Forest Pipeline
    print("\n=== Running Random Forest Pipeline ===")
    # This will trigger feature generation (since cache is empty) and then train RF.
    rf_results = model_rf.run_rf_pipeline(load_cached_data=False)

    # Validation of RF Results
    assert "val_preds" in rf_results, "RF results missing validation predictions"
    assert "test_preds" in rf_results, "RF results missing test predictions"
    assert "val_auc" in rf_results, "RF results missing validation AUC"
    assert isinstance(rf_results["val_auc"], float), "RF AUC must be a float"
    # AUC might be 0.0 if validation set is empty or something fails, but usually > 0.5
    assert 0.0 <= rf_results["val_auc"] <= 1.0, "RF AUC out of range"
    assert len(rf_results["test_preds"]) == 1162, "RF Test predictions count mismatch"

    print(f"RF Validation AUC: {rf_results['val_auc']:.4f}")

    # 4. MLP Pipeline
    print("\n=== Running MLP Pipeline ===")
    # The features were computed and cached during the RF step.
    # This step will load them from disk and train the Neural Network.
    mlp_results = model_mlp.run_mlp_pipeline(load_cached_data=True)

    # Validation of MLP Results
    assert "val_preds" in mlp_results, "MLP results missing validation predictions"
    assert "test_preds" in mlp_results, "MLP results missing test predictions"
    assert "val_auc" in mlp_results, "MLP results missing validation AUC"
    assert isinstance(mlp_results["val_auc"], float), "MLP AUC must be a float"
    assert 0.0 <= mlp_results["val_auc"] <= 1.0, "MLP AUC out of range"
    assert len(mlp_results["test_preds"]) == 1162, "MLP Test predictions count mismatch"

    print(f"MLP Validation AUC: {mlp_results['val_auc']:.4f}")

    # 5. Ensemble and Submission
    print("\n=== Generating Submission ===")

    # Simple weighted average ensemble defined in config
    rf_weight = config.ENSEMBLE_WEIGHTS["rf"]
    mlp_weight = config.ENSEMBLE_WEIGHTS["mlp"]

    print(f"Ensembling with weights -> RF: {rf_weight}, MLP: {mlp_weight}")
    final_preds = (rf_results["test_preds"] * rf_weight) + (
        mlp_results["test_preds"] * mlp_weight
    )

    # Load Test Metadata to get Request IDs
    # We use the metadata file generated in the setup phase
    test_df = pd.read_csv(config.TEST_PATH)
    assert len(test_df) == len(
        final_preds
    ), "Mismatch between test data rows and predictions"

    # Create Submission DataFrame
    submission = pd.DataFrame(
        {config.ID_COL: test_df[config.ID_COL], config.TARGET_COL: final_preds}
    )

    # Save to disk
    submission_path = "./submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    # Final Verification
    assert os.path.exists(submission_path), "Submission file was not created"

    saved_df = pd.read_csv(submission_path)
    assert saved_df.shape == (1162, 2), f"Submission shape mismatch: {saved_df.shape}"
    assert config.ID_COL in saved_df.columns, "Missing ID column in submission"
    assert config.TARGET_COL in saved_df.columns, "Missing Target column in submission"

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
