import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library import config
from library import data_loader
from library import feature_engineering
from library import model_rf
from library import model_mlp


def run_demo():
    print("Starting execution of library demo...")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # -------------------------------------------------------------------------
    print("\n[1] Configuring for rapid execution...")

    # Reduce sample size for debugging
    config.DEBUG_SAMPLE_SIZE = 50

    # Reduce Random Forest complexity
    config.RF_PARAMS["n_estimators"] = 10
    config.RF_PARAMS["n_jobs"] = 1  # Avoid overhead for small data

    # Reduce MLP complexity
    config.MLP_PARAMS["epochs"] = 2
    config.MLP_PARAMS["batch_size"] = 8
    config.MLP_PARAMS["hidden_dim"] = 32  # Smaller network
    config.MLP_PARAMS["patience"] = 1

    # Ensure reproducibility
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    # -------------------------------------------------------------------------
    # 2. Data Loading Demo
    # -------------------------------------------------------------------------
    print("\n[2] Demonstrating Data Loader...")

    # Load data in debug mode (subsampled)
    # We force load_cached_data=False to ensure we test the raw loading logic once
    train_df, val_df, test_df = data_loader.load_dataset(
        load_cached_data=False, debug=True
    )

    print(f"  Train shape: {train_df.shape}")
    print(f"  Val shape: {val_df.shape}")
    print(f"  Test shape: {test_df.shape}")

    # Assertions
    assert (
        len(train_df) == config.DEBUG_SAMPLE_SIZE
    ), f"Train size mismatch: {len(train_df)}"
    assert "requester_received_pizza" in train_df.columns, "Target missing in train"
    assert "requester_received_pizza" not in test_df.columns, "Target leaked to test"

    # -------------------------------------------------------------------------
    # 3. Feature Engineering Demo
    # -------------------------------------------------------------------------
    print("\n[3] Demonstrating Feature Engineering...")

    fe = feature_engineering.FeatureEngineer()

    # A. Metadata Features
    print("  Generating Metadata Features...")
    meta_df = fe.generate_metadata_features(train_df)
    assert not meta_df.empty, "Metadata dataframe is empty"
    assert "title_len_char" in meta_df.columns, "Text meta-feature missing"

    # B. Zero-Shot Profiles
    print("  Generating Zero-Shot Action Profiles...")
    # Using 'train' split name for cache handling
    profiles_df = fe.generate_zero_shot_profiles(
        train_df, "train_debug", load_cached_data=False
    )
    assert profiles_df.shape[0] == len(train_df), "Profile rows mismatch"
    assert profiles_df.shape[1] == len(
        config.SEMANTIC_ANCHORS
    ), "Profile columns mismatch"

    # C. TF-IDF Features
    print("  Generating TF-IDF Features...")
    tfidf_train, tfidf_val, tfidf_test = fe.generate_tfidf_features(
        train_df, val_df, test_df, load_cached_data=False
    )
    assert tfidf_train.shape[0] == len(train_df), "TF-IDF train rows mismatch"

    # -------------------------------------------------------------------------
    # 4. Stream A: Random Forest Model Demo
    # -------------------------------------------------------------------------
    print("\n[4] Demonstrating Random Forest Pipeline...")

    # Train RF model using the library function (handles assembly internally)
    rf_results = model_rf.train_rf_model(load_cached_data=False, debug=True)

    # Verify results
    assert "model" in rf_results
    assert "val_probs" in rf_results
    assert "test_probs" in rf_results
    assert 0 <= rf_results["auc"] <= 1, f"Invalid RF AUC: {rf_results['auc']}"
    assert len(rf_results["test_probs"]) == len(
        test_df
    ), "RF test predictions length mismatch"

    print(f"  RF Validation AUC: {rf_results['auc']:.4f}")

    # -------------------------------------------------------------------------
    # 5. Stream B: MLP Model Demo
    # -------------------------------------------------------------------------
    print("\n[5] Demonstrating MLP Pipeline...")

    # Train MLP model using the library function
    mlp_results = model_mlp.train_mlp_model(load_cached_data=False, debug=True)

    # Verify results
    assert "model" in mlp_results
    assert "val_probs" in mlp_results
    assert "test_probs" in mlp_results
    assert 0 <= mlp_results["auc"] <= 1, f"Invalid MLP AUC: {mlp_results['auc']}"
    assert len(mlp_results["test_probs"]) == len(
        test_df
    ), "MLP test predictions length mismatch"

    print(f"  MLP Validation AUC: {mlp_results['auc']:.4f}")

    # -------------------------------------------------------------------------
    # 6. Ensemble Demo
    # -------------------------------------------------------------------------
    print("\n[6] Demonstrating Ensemble Strategy...")

    # Weights
    w_rf = config.ENSEMBLE_WEIGHTS["rf"]
    w_mlp = config.ENSEMBLE_WEIGHTS["mlp"]

    # Combine Validation Predictions
    # Note: y_val should be consistent across both streams as they use the same data split
    y_val = rf_results["y_val"]
    val_preds_ensemble = (w_rf * rf_results["val_probs"]) + (
        w_mlp * mlp_results["val_probs"]
    )

    ensemble_auc = roc_auc_score(y_val, val_preds_ensemble)
    print(f"  Ensemble Validation AUC: {ensemble_auc:.4f}")

    # Combine Test Predictions
    test_preds_ensemble = (w_rf * rf_results["test_probs"]) + (
        w_mlp * mlp_results["test_probs"]
    )

    # -------------------------------------------------------------------------
    # 7. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[7] Generating Submission File...")

    submission_df = pd.DataFrame(
        {
            "request_id": test_df["request_id"],
            "requester_received_pizza": test_preds_ensemble,
        }
    )

    # Ensure output directory exists
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # Save
    submission_path = config.SUBMISSION_PATH
    submission_df.to_csv(submission_path, index=False)

    print(f"  Submission saved to: {submission_path}")

    # Verify file
    assert os.path.exists(submission_path), "Submission file not created"
    loaded_sub = pd.read_csv(submission_path)
    assert len(loaded_sub) == len(test_df), "Submission row count mismatch"
    assert "request_id" in loaded_sub.columns
    assert "requester_received_pizza" in loaded_sub.columns

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
