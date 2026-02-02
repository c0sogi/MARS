import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, ensure_dir
from library.data_loader import DataLoader
from library.features import FeatureGenerator
from library.rf_stream import train_rf_model, predict_rf
from library.mlp_stream import train_mlp_model, predict_mlp


def run_demo():
    # 1. Setup and Configuration Override for Speed
    print("1. Setting up configuration for fast demonstration...")
    set_seed(Config.RANDOM_STATE)

    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 100  # Small subset for demo
    Config.RF_N_ESTIMATORS = 10  # Fewer trees
    Config.MLP_EPOCHS = 2  # Fewer epochs
    Config.MLP_BATCH_SIZE = 16
    Config.MLP_PATIENCE = 1  # Early stopping check

    # Ensure working directories exist
    ensure_dir(Config.WORKING_DIR)
    ensure_dir(Config.SUBMISSION_DIR)

    Config.print_config()

    # 2. Data Loading
    print("\n2. Loading Data...")
    # Force reload to ensure we use the debug subset logic
    df_train, df_val, df_test = DataLoader.load_data(load_cached_data=False)

    # Verification
    print(f"  Train shape: {df_train.shape}")
    print(f"  Val shape: {df_val.shape}")
    print(f"  Test shape: {df_test.shape}")

    assert (
        len(df_train) <= Config.DEBUG_SAMPLE_SIZE
    ), "Train set size exceeds debug limit"
    assert (
        "requester_received_pizza" in df_train.columns
    ), "Target column missing in train"
    assert (
        "requester_received_pizza" not in df_test.columns
    ), "Target column should not be in test"

    # 3. Feature Generation
    print("\n3. Generating Features...")
    feature_gen = FeatureGenerator()
    # Force regeneration to match the debug data subset
    rf_features, mlp_features = feature_gen.process_data(
        df_train, df_val, df_test, load_cached_data=False
    )

    # Verification of RF Features
    assert "X_train" in rf_features and "y_train" in rf_features
    assert "X_test" in rf_features
    assert rf_features["X_train"].shape[0] == len(df_train)

    # Verification of MLP Features
    # Check for a specific key pattern expected by PizzaDataset
    assert "train_title_emb" in mlp_features
    assert "test_metadata" in mlp_features
    assert mlp_features["train_title_emb"].shape[0] == len(df_train)

    # 4. Stream A: Random Forest
    print("\n4. Running Stream A (Random Forest)...")
    rf_model_path = os.path.join(Config.WORKING_DIR, "rf_model_demo.pkl")

    rf_model = train_rf_model(
        X_train=rf_features["X_train"],
        y_train=rf_features["y_train"],
        X_val=rf_features["X_val"],
        y_val=rf_features["y_val"],
        save_path=rf_model_path,
    )

    # Predict
    rf_preds_test = predict_rf(rf_model, rf_features["X_test"])

    # Verify predictions
    assert len(rf_preds_test) == len(df_test)
    assert np.all(
        (rf_preds_test >= 0) & (rf_preds_test <= 1)
    ), "RF predictions out of probability range"
    print(f"  RF Test Predictions Mean: {np.mean(rf_preds_test):.4f}")

    # 5. Stream B: MLP
    print("\n5. Running Stream B (MLP)...")
    mlp_model_path = os.path.join(Config.WORKING_DIR, "mlp_model_demo.pth")

    mlp_model = train_mlp_model(features_dict=mlp_features, save_path=mlp_model_path)

    # Predict
    mlp_preds_test = predict_mlp(mlp_model, mlp_features, split="test")

    # Verify predictions
    assert len(mlp_preds_test) == len(df_test)
    assert np.all(
        (mlp_preds_test >= 0) & (mlp_preds_test <= 1)
    ), "MLP predictions out of probability range"
    print(f"  MLP Test Predictions Mean: {np.mean(mlp_preds_test):.4f}")

    # 6. Ensemble
    print("\n6. Ensembling...")
    final_preds = (
        Config.ENSEMBLE_WEIGHT_RF * rf_preds_test
        + Config.ENSEMBLE_WEIGHT_MLP * mlp_preds_test
    )

    # Verify Ensemble
    assert len(final_preds) == len(df_test)

    # 7. Submission
    print("\n7. Creating Submission...")
    submission = pd.DataFrame(
        {"request_id": df_test["request_id"], "requester_received_pizza": final_preds}
    )

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission_demo.csv")
    submission.to_csv(submission_path, index=False)

    print(f"  Submission saved to: {submission_path}")
    print("  Head of submission:")
    print(submission.head())

    # Final Validation of Submission File
    loaded_sub = pd.read_csv(submission_path)
    assert loaded_sub.shape == (len(df_test), 2)
    assert list(loaded_sub.columns) == ["request_id", "requester_received_pizza"]

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
