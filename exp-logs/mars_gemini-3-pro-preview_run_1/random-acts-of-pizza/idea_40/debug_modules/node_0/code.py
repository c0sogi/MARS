import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch

# Import library modules
from library import config, utils, features, dataset, mlp_model, rf_model, training


def run_demo():
    print("Starting Demo Script...")

    # 0. Setup and Configuration Overrides for Speed
    warnings.filterwarnings("ignore")
    utils.set_seed(42)

    # Force Debug mode globally for this script to use small data subsets (N=100)
    config.DEBUG = True
    config.DEBUG_SAMPLE_SIZE = 100

    # Reduce model complexity for rapid execution
    config.RF_PARAMS["n_estimators"] = 5
    config.RF_PARAMS["n_jobs"] = 1  # Avoid overhead in debug

    config.MLP_PARAMS["epochs"] = 1
    config.MLP_PARAMS["batch_size"] = 16
    config.MLP_PARAMS["hidden_dim_fusion"] = 32  # Smaller for debug

    print(
        f"Configuration set: DEBUG={config.DEBUG}, Sample Size={config.DEBUG_SAMPLE_SIZE}"
    )

    # 1. Verify Data Loading Utils
    print("\n--- Testing Utils ---")
    df_train, df_val, df_test = utils.load_data(debug=True)

    print(f"Train shape: {df_train.shape}")
    print(f"Val shape: {df_val.shape}")
    print(f"Test shape: {df_test.shape}")

    assert len(df_train) == config.DEBUG_SAMPLE_SIZE, "Train set size mismatch"
    assert len(df_val) == config.DEBUG_SAMPLE_SIZE, "Val set size mismatch"
    assert len(df_test) == config.DEBUG_SAMPLE_SIZE, "Test set size mismatch"

    common_cols = utils.get_common_columns(df_train, df_test)
    assert len(common_cols) > 0, "No common columns found"
    print("Utils verification passed.")

    # 2. Verify Feature Engineering
    print("\n--- Testing Feature Engineering ---")
    fe = features.FeatureEngineer()

    # Force re-computation (load_cached_data=False) to ensure logic runs
    # Note: Since config.DEBUG is True, this processes the small subset
    rf_out, mlp_out = fe.process_data(load_cached_data=False)

    # Check RF Feature Shapes
    assert rf_out["X_train"].shape[0] == config.DEBUG_SAMPLE_SIZE
    assert rf_out["y_train"].shape[0] == config.DEBUG_SAMPLE_SIZE
    print(f"RF Features generated. X_train shape: {rf_out['X_train'].shape}")

    # Check MLP Feature Shapes
    assert mlp_out["train_title_emb"].shape == (config.DEBUG_SAMPLE_SIZE, 384)
    assert mlp_out["train_hist_seq"].shape == (config.DEBUG_SAMPLE_SIZE, 20, 384)
    print(
        f"MLP Features generated. Title Emb shape: {mlp_out['train_title_emb'].shape}"
    )
    print("Feature Engineering verification passed.")

    # 3. Verify Random Forest Pipeline
    print("\n--- Testing RF Pipeline ---")
    # run_rf_pipeline handles feature loading internally.
    # We pass debug=True to ensure it uses the debug configuration logic.
    rf_results = rf_model.run_rf_pipeline(load_cached_data=True, debug=True)

    assert "model" in rf_results
    assert "val_auc" in rf_results
    assert "test_preds" in rf_results

    val_auc = rf_results["val_auc"]
    test_preds = rf_results["test_preds"]

    print(f"RF Validation AUC: {val_auc}")
    assert 0.0 <= val_auc <= 1.0, "Invalid AUC score"
    assert len(test_preds) == config.DEBUG_SAMPLE_SIZE, "Prediction count mismatch"
    print("RF Pipeline verification passed.")

    # 4. Verify MLP Model Architecture
    print("\n--- Testing MLP Model Architecture ---")
    # Get metadata dimension from generated features
    meta_dim = mlp_out["train_meta"].shape[1]

    # Instantiate model
    model = mlp_model.SkipGatedDualQueryMLP(input_meta_dim=meta_dim)

    # Create dummy batch
    batch_size = 4
    dummy_title = torch.randn(batch_size, 384)
    dummy_body = torch.randn(batch_size, 384)
    dummy_hist = torch.randn(batch_size, 20, 384)
    dummy_mask = torch.ones(batch_size, 20)
    dummy_meta = torch.randn(batch_size, meta_dim)
    dummy_cons = torch.randn(batch_size, 2)

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(
            dummy_title, dummy_body, dummy_hist, dummy_mask, dummy_meta, dummy_cons
        )

    assert logits.shape == (batch_size,), f"Output shape mismatch: {logits.shape}"
    print("MLP Architecture forward pass successful.")

    # 5. Verify MLP Training Pipeline
    print("\n--- Testing MLP Training Pipeline ---")
    # This runs the full training loop for 1 epoch on the debug dataset
    mlp_results = training.train_mlp_model(load_cached_data=True, debug=True)

    assert "model" in mlp_results
    assert "val_auc" in mlp_results
    assert "test_preds" in mlp_results

    mlp_preds = mlp_results["test_preds"]
    assert len(mlp_preds) == config.DEBUG_SAMPLE_SIZE, "MLP Prediction count mismatch"
    print(f"MLP Validation AUC: {mlp_results['val_auc']}")
    print("MLP Training Pipeline verification passed.")

    # 6. Verify Submission Generation
    print("\n--- Testing Submission Generation ---")
    request_ids = mlp_results["request_ids"]
    output_path = "./working/demo_submission.csv"

    utils.save_submission(request_ids, mlp_preds, output_path=output_path)

    assert os.path.exists(output_path), "Submission file not created"

    # Check file content
    df_sub = pd.read_csv(output_path)
    assert list(df_sub.columns) == ["request_id", "requester_received_pizza"]
    assert len(df_sub) == config.DEBUG_SAMPLE_SIZE
    print(f"Submission saved to {output_path}")
    print("Submission verification passed.")

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    run_demo()
