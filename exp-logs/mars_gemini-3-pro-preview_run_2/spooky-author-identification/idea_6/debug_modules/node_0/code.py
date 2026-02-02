import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.utils import (
    seed_everything,
    calculate_metric,
    save_artifact,
    load_artifact,
)
from library.data_factory import load_train_data, load_test_data, LABEL_MAP
from library.feature_engineering import get_classical_features
from library.training_engine import train_classical_fold, train_neural_fold
from library.stacking import StackingEnsemble

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Library Usage Demo ===\n")

    # ---------------------------------------------------------
    # 1. Configuration Setup for Fast Execution
    # ---------------------------------------------------------
    print("1. Configuring environment for fast demo execution...")

    # Override Config defaults to run quickly on a small subset
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Very small sample for speed
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 8
    Config.GRADIENT_ACCUMULATION_STEPS = 1
    Config.SVD_COMPONENTS = 5  # Reduce components for speed
    Config.MIN_DF = 1  # Allow rare words in small sample
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Re-run setup to create new directories
    Config.setup()

    # Set seed
    seed_everything(Config.SEED)
    print("   Configuration updated. Working directory:", Config.WORKING_DIR)

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("\n2. Loading Data...")

    # Load Train Data (Debug mode returns a small sample)
    train_df_full = load_train_data(load_cached_data=False, debug=True)

    # Load Test Data (Slice manually to match debug size)
    test_df = load_test_data().head(Config.DEBUG_SAMPLE_SIZE).copy()

    print(f"   Train Data Shape: {train_df_full.shape}")
    print(f"   Test Data Shape: {test_df.shape}")

    # Validate Data
    assert not train_df_full.empty, "Training dataframe is empty"
    assert "text" in train_df_full.columns and "author" in train_df_full.columns
    assert "fold" in train_df_full.columns

    # ---------------------------------------------------------
    # 3. Feature Engineering
    # ---------------------------------------------------------
    print("\n3. Generating Classical Features (TF-IDF & SVD)...")

    # This function saves artifacts to disk and returns them
    train_tfidf, test_tfidf, train_svd, test_svd, train_y_full = get_classical_features(
        train_df_full, test_df, load_cached_data=False
    )

    print(f"   TF-IDF Train Shape: {train_tfidf.shape}")
    print(f"   SVD Train Shape: {train_svd.shape}")

    # Validate Features
    assert train_tfidf.shape[0] == len(train_df_full)
    assert train_svd.shape[1] == Config.SVD_COMPONENTS
    assert train_y_full.shape[0] == len(train_df_full)

    # ---------------------------------------------------------
    # 4. Simulating a Single Fold Split
    # ---------------------------------------------------------
    print("\n4. Preparing Single Fold Split...")

    # For this demo, we manually split the debug data into "Train" and "Validation"
    # In a real run, this is handled by the 'fold' column and a loop.
    split_idx = int(len(train_df_full) * 0.8)

    # Split DataFrames
    df_train = train_df_full.iloc[:split_idx].reset_index(drop=True)
    df_val = train_df_full.iloc[split_idx:].reset_index(drop=True)

    # Split Features (Sparse and Dense)
    X_tfidf_train = train_tfidf[:split_idx]
    X_tfidf_val = train_tfidf[split_idx:]

    X_svd_train = train_svd[:split_idx]
    X_svd_val = train_svd[split_idx:]

    y_train = train_y_full[:split_idx]
    y_val = train_y_full[split_idx:]

    print(f"   Split sizes - Train: {len(df_train)}, Val: {len(df_val)}")

    # ---------------------------------------------------------
    # 5. Training Classical Models
    # ---------------------------------------------------------
    print("\n5. Training Classical Models (LR, NB, XGB)...")

    classical_results = train_classical_fold(
        fold_idx=0,
        X_tfidf_train=X_tfidf_train,
        y_train=y_train,
        X_tfidf_val=X_tfidf_val,
        y_val=y_val,
        X_tfidf_test=test_tfidf,
        X_svd_train=X_svd_train,
        X_svd_val=X_svd_val,
        X_svd_test=test_svd,
    )

    # Validate Classical Results
    for model_name in ["lr", "nb", "xgb"]:
        assert model_name in classical_results
        assert classical_results[model_name]["val"].shape == (len(df_val), 3)
        assert classical_results[model_name]["test"].shape == (len(test_df), 3)
        print(f"   {model_name.upper()} training complete.")

    # ---------------------------------------------------------
    # 6. Training Neural Model
    # ---------------------------------------------------------
    print("\n6. Training Neural Model (DeBERTa)...")

    # This uses the DataFrames directly
    oof_neural, test_neural = train_neural_fold(
        fold_idx=0, train_df=df_train, val_df=df_val, test_df=test_df
    )

    # Validate Neural Results
    assert oof_neural.shape == (len(df_val), 3)
    assert test_neural.shape == (len(test_df), 3)
    print("   Neural training complete.")

    # ---------------------------------------------------------
    # 7. Stacking Ensemble
    # ---------------------------------------------------------
    print("\n7. Running Stacking Ensemble...")

    stacker = StackingEnsemble()

    # Collect OOF predictions (Validation set of the current fold)
    oof_preds_dict = {
        "lr": classical_results["lr"]["val"],
        "nb": classical_results["nb"]["val"],
        "xgb": classical_results["xgb"]["val"],
        "neural": oof_neural,
    }

    # Collect Test predictions
    test_preds_dict = {
        "lr": classical_results["lr"]["test"],
        "nb": classical_results["nb"]["test"],
        "xgb": classical_results["xgb"]["test"],
        "neural": test_neural,
    }

    # Train meta-learner and predict
    # Note: y_val contains the ground truth for the OOF predictions
    final_test_preds = stacker.fit_predict(
        oof_preds_dict=oof_preds_dict,
        test_preds_dict=test_preds_dict,
        y_train=y_val,
        test_ids=test_df["id"].values,
    )

    # ---------------------------------------------------------
    # 8. Final Verification
    # ---------------------------------------------------------
    print("\n8. Verifying Output...")

    # Check if submission file exists
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Load and check content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert sub_df.shape == (
        len(test_df),
        4,
    ), f"Submission shape mismatch: {sub_df.shape}"
    assert list(sub_df.columns) == [
        "id",
        "EAP",
        "HPL",
        "MWS",
    ], "Submission columns mismatch"

    # Check probability constraints
    probs = sub_df[["EAP", "HPL", "MWS"]].values
    assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities out of bounds"

    print("   Submission file verified successfully.")

    # Demonstrate Artifact Utils
    print("\n9. Demonstrating Artifact Utilities...")
    dummy_data = np.array([1, 2, 3])
    save_artifact(dummy_data, "demo_artifact.npy")
    loaded_data = load_artifact("demo_artifact.npy")
    assert np.array_equal(dummy_data, loaded_data)
    print("   Artifact save/load verified.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
