import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import the provided library modules
from library import config, data_loader, feature_engine, neural_net, train_eval

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Library Usage Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Debugging
    # -------------------------------------------------------------------------
    print("[1] Configuring environment for fast demonstration...")

    # Enable debug mode to limit data size in data_loader
    config.DEBUG = True
    config.MAX_SAMPLES = 50  # Only load 50 samples per split

    # Reduce Model Complexity for speed
    config.RF_N_ESTIMATORS = 5
    config.MLP_EPOCHS = 1
    config.MLP_BATCH_SIZE = 8
    config.MLP_HIDDEN_DIM = 16
    config.SBERT_MODEL = "all-MiniLM-L6-v2"  # Ensure small model is used

    # Set a custom working directory for this demo to avoid conflicts
    config.WORKING_DIR = "./working/demo_execution"
    config.SUBMISSION_PATH = "./working/demo_output/demo_submission.csv"

    os.makedirs(config.WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

    print(f"    Debug Mode: {config.DEBUG}")
    print(f"    Max Samples: {config.MAX_SAMPLES}")
    print(f"    Working Directory: {config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n[2] Demonstrating Data Loading...")

    # Force reload from source (ignoring potential previous caches) to ensure we get the small subset
    df_train, df_val = data_loader.get_stratified_split(load_cached_data=False)
    df_test = data_loader.load_dataset("test", load_cached_data=False)

    print(f"    Train shape: {df_train.shape}")
    print(f"    Val shape:   {df_val.shape}")
    print(f"    Test shape:  {df_test.shape}")

    # Verification
    assert len(df_train) <= config.MAX_SAMPLES, "Train set size exceeds debug limit."
    assert len(df_val) <= config.MAX_SAMPLES, "Val set size exceeds debug limit."
    assert "requester_subreddits_at_request" in df_train.columns, "List column missing."
    assert isinstance(
        df_train.iloc[0]["requester_subreddits_at_request"], list
    ), "List column not parsed correctly."
    print("    Data loading and parsing verified.")

    # -------------------------------------------------------------------------
    # 3. Feature Engineering Demonstration
    # -------------------------------------------------------------------------
    print("\n[3] Demonstrating Feature Engineering...")

    # We use the pipeline with load_cached_data=False to force computation on our small debug sets
    # This returns: (X_rf, data_nn) tuples for train, val, test
    train_res, val_res, test_res = feature_engine.run_feature_pipeline(
        load_cached_data=False
    )

    X_rf_train, data_nn_train = train_res
    X_rf_val, data_nn_val = val_res
    X_rf_test, data_nn_test = test_res

    # Verification of Random Forest Features (Dense Numpy Array)
    print(f"    RF Train Features Shape: {X_rf_train.shape}")
    assert isinstance(X_rf_train, np.ndarray), "RF features must be a numpy array."
    assert X_rf_train.shape[0] == len(
        df_train
    ), "RF feature rows must match dataframe rows."
    assert not np.isnan(X_rf_train).any(), "RF features contain NaNs."

    # Verification of Neural Network Data (Dictionary of Arrays)
    print("    NN Train Data Keys:", list(data_nn_train.keys()))
    assert "title_emb" in data_nn_train
    assert "meta" in data_nn_train
    assert data_nn_train["title_emb"].shape[0] == len(
        df_train
    ), "NN embeddings rows must match dataframe rows."

    # Check labels existence
    assert "labels" in data_nn_train, "Labels missing from NN train data."
    assert "labels" not in data_nn_test, "Labels should not be in NN test data."

    print("    Feature engineering verified.")

    # -------------------------------------------------------------------------
    # 4. Model Training Demonstration
    # -------------------------------------------------------------------------
    print("\n[4] Demonstrating Model Training...")

    # Extract labels
    y_train = data_nn_train["labels"]
    y_val = data_nn_val["labels"]

    # A. Random Forest (Stream A)
    print("    Training Random Forest...")
    rf_model = train_eval.train_rf(X_rf_train, y_train)

    # Verify RF
    assert hasattr(rf_model, "predict_proba"), "RF model not fitted correctly."
    rf_preds = rf_model.predict_proba(X_rf_val)[:, 1]
    assert len(rf_preds) == len(y_val), "RF prediction shape mismatch."
    print("    Random Forest trained successfully.")

    # B. Neural Network (Stream B)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    Training Neural Network on {device}...")

    # train_nn returns (model, best_auc)
    nn_model, nn_auc = train_eval.train_nn(data_nn_train, data_nn_val, device)

    # Verify NN
    assert isinstance(nn_model, torch.nn.Module), "NN model is not a PyTorch module."
    assert 0 <= nn_auc <= 1, "NN AUC score out of bounds."
    print(f"    Neural Network trained successfully. Best Val AUC: {nn_auc:.4f}")

    # -------------------------------------------------------------------------
    # 5. Evaluation & Submission Demonstration
    # -------------------------------------------------------------------------
    print("\n[5] Demonstrating Evaluation and Submission...")

    # Evaluate Ensemble
    ensemble_auc = train_eval.evaluate_ensemble(
        rf_model, nn_model, X_rf_val, data_nn_val, y_val, device
    )
    assert 0 <= ensemble_auc <= 1, "Ensemble AUC out of bounds."

    # Generate Submission
    train_eval.generate_submission(rf_model, nn_model, X_rf_test, data_nn_test, device)

    # Verify Output
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not created."

    submission_df = pd.read_csv(config.SUBMISSION_PATH)
    print(f"    Submission loaded. Shape: {submission_df.shape}")
    print("    Head:\n", submission_df.head(3))

    assert list(submission_df.columns) == [
        "request_id",
        "requester_received_pizza",
    ], "Invalid submission columns."
    assert len(submission_df) == len(df_test), "Submission row count mismatch."
    assert (
        submission_df["requester_received_pizza"].between(0, 1).all()
    ), "Probabilities out of bounds."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
