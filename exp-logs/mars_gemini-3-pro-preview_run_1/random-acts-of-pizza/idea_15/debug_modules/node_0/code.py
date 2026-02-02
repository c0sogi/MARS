import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Import the provided library modules
from library import config
from library import data_loader
from library import features
from library import model_rf
from library import model_mlp


def run_demo():
    print("Initializing Demo Execution...")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # -------------------------------------------------------------------------
    print("Overriding configuration for fast demonstration...")
    # Reduce Random Forest complexity
    config.RF_PARAMS["n_estimators"] = 5
    config.RF_PARAMS["n_jobs"] = 1  # Avoid overhead for small demo

    # Reduce MLP complexity
    config.MLP_PARAMS["epochs"] = 1
    config.MLP_PARAMS["hidden_dim"] = 32
    config.MLP_PARAMS["batch_size"] = 16

    # Reduce Feature complexity
    config.TFIDF_MAX_FEATURES = 50

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Testing Data Loader ---")
    # We force load_cached_data=False to demonstrate the parsing logic
    train_df, val_df, test_df = data_loader.load_datasets(load_cached_data=False)

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    # Validation
    assert not train_df.empty, "Training dataframe is empty."
    assert (
        "requester_received_pizza" in train_df.columns
    ), "Target column missing in train."
    assert isinstance(
        train_df["requester_subreddits_at_request"].iloc[0], list
    ), "List column parsing failed."

    # -------------------------------------------------------------------------
    # 3. Feature Engineering Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Testing Feature Engineer ---")
    fe = features.FeatureEngineer()

    # Test Metadata Generation (Subset for speed)
    print("Generating metadata features (subset)...")
    meta_df = fe.generate_metadata_features(
        train_df.head(50), "train_debug", load_cached_data=False
    )
    assert "upvote_ratio" in meta_df.columns, "Ratio feature engineering failed."
    assert "body_len_chars_arcsinh" in meta_df.columns, "Arcsinh transformation failed."

    # Test TF-IDF (Full set required for consistency across splits)
    print("Generating TF-IDF features...")
    # We use the actual dataframes here
    tfidf_out = fe.get_tfidf_features(train_df, val_df, test_df, load_cached_data=False)
    train_title_tfidf = tfidf_out[0]
    assert (
        train_title_tfidf.shape[1] <= config.TFIDF_MAX_FEATURES
    ), "TF-IDF max features not respected."

    # Note: We skip explicit SBERT calls here as they are integrated into the model streams below
    # and calling them twice would waste time in this demo.

    # -------------------------------------------------------------------------
    # 4. Model Stream A: Random Forest
    # -------------------------------------------------------------------------
    print("\n--- Testing Random Forest Stream ---")
    # This function internally calls feature generation (SBERT, Anchors, etc.)
    rf_model, rf_val_probs, rf_test_probs = model_rf.train_rf_stream(
        load_cached_data=False
    )

    # Validation
    assert len(rf_val_probs) == len(val_df), "RF val predictions length mismatch."
    assert len(rf_test_probs) == len(test_df), "RF test predictions length mismatch."
    assert (
        0.0 <= np.min(rf_val_probs) and np.max(rf_val_probs) <= 1.0
    ), "RF probabilities out of bounds."

    rf_auc = roc_auc_score(val_df["requester_received_pizza"].astype(int), rf_val_probs)
    print(f"Demo RF Validation AUC: {rf_auc:.4f}")

    # -------------------------------------------------------------------------
    # 5. Model Stream B: MLP
    # -------------------------------------------------------------------------
    print("\n--- Testing MLP Stream ---")
    # This function prepares tensors, history sequences, and trains the Residual Attention Net
    mlp_model, mlp_val_probs, mlp_test_probs = model_mlp.train_mlp_stream(
        load_cached_data=False
    )

    # Validation
    assert len(mlp_val_probs) == len(val_df), "MLP val predictions length mismatch."
    assert len(mlp_test_probs) == len(test_df), "MLP test predictions length mismatch."

    mlp_auc = roc_auc_score(
        val_df["requester_received_pizza"].astype(int), mlp_val_probs
    )
    print(f"Demo MLP Validation AUC: {mlp_auc:.4f}")

    # -------------------------------------------------------------------------
    # 6. Ensemble and Submission
    # -------------------------------------------------------------------------
    print("\n--- Testing Ensemble ---")
    w_rf = config.ENSEMBLE_WEIGHTS["rf"]
    w_mlp = config.ENSEMBLE_WEIGHTS["mlp"]

    final_test_probs = (w_rf * rf_test_probs) + (w_mlp * mlp_test_probs)

    submission_df = pd.DataFrame(
        {
            "request_id": test_df["request_id"],
            "requester_received_pizza": final_test_probs,
        }
    )

    print("Sample Submission:")
    print(submission_df.head())

    # Save dummy submission
    out_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")

    assert os.path.exists(out_path), "Submission file not created."

    print("\nDemo Execution Completed Successfully.")


if __name__ == "__main__":
    # Set seeds for reproducibility of the demo script itself
    np.random.seed(42)
    torch.manual_seed(42)
    run_demo()
