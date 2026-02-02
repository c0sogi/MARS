import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data_loader import load_data
from library.feature_engine import FeatureExtractor
from library.rf_model import RFModel
from library.mlp_model import MLPModel


def run_demo():
    print("Initializing Demo Execution...")

    # 1. Monkey-patch Config for Speed
    # We override specific settings to ensure the demo runs quickly (within minutes)
    # while still exercising all code paths.
    print("Configuring runtime parameters for speed...")
    Config.WORKING_DIR = "./working/demo_execution"
    Config.RF_N_ESTIMATORS = 10  # Reduced from 500
    Config.MLP_EPOCHS = 2  # Reduced from 50
    Config.MLP_BATCH_SIZE = 32
    Config.PCA_COMPONENTS = 10  # Reduced for speed

    # Ensure demo working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.RANDOM_STATE)

    # 2. Feature Engineering
    # We initialize the feature extractor and run it.
    # We set load_cached_data=False to force the computation of features
    # to demonstrate the engineering pipeline.
    print("\n--- Step 1: Feature Engineering ---")
    extractor = FeatureExtractor()

    # This will load data via data_loader.load_data internally
    feature_data = extractor.run(load_cached_data=False)

    # Verify Feature Data Structure
    print("Verifying feature data integrity...")
    for split in ["train", "val", "test"]:
        assert split in feature_data, f"Missing {split} in feature data"
        assert "rf_features" in feature_data[split], f"Missing RF features for {split}"
        assert "mlp" in feature_data[split], f"Missing MLP features for {split}"

        # Check MLP sub-features
        mlp_feats = feature_data[split]["mlp"]
        assert "request_emb" in mlp_feats
        assert "history_seq" in mlp_feats
        assert "metadata" in mlp_feats

        # Check Shapes (Basic check)
        n_samples = mlp_feats["request_emb"].shape[0]
        assert mlp_feats["history_seq"].shape[0] == n_samples
        assert mlp_feats["metadata"].shape[0] == n_samples

        if split != "test":
            assert "y" in feature_data[split]
            assert len(feature_data[split]["y"]) == n_samples

    print("Feature extraction successful.")

    # 3. Stream A: Random Forest Model
    print("\n--- Step 2: Training Random Forest (Stream A) ---")
    rf_model = RFModel()

    # Train
    rf_model.train(
        X_train=feature_data["train"]["rf_features"],
        y_train=feature_data["train"]["y"],
        X_val=feature_data["val"]["rf_features"],
        y_val=feature_data["val"]["y"],
    )

    # Predict on Test
    print("Generating RF predictions...")
    rf_preds_test = rf_model.predict_proba(feature_data["test"]["rf_features"])

    # Verify Predictions
    assert len(rf_preds_test) == feature_data["test"]["rf_features"].shape[0]
    assert np.all((rf_preds_test >= 0) & (rf_preds_test <= 1))
    print(f"RF Test Predictions generated. Shape: {rf_preds_test.shape}")

    # 4. Stream B: MLP Model
    print("\n--- Step 3: Training MLP (Stream B) ---")
    mlp_model = MLPModel()

    # Train
    # The train method expects dictionaries containing 'mlp' features and 'y'
    mlp_model.train(data_train=feature_data["train"], data_val=feature_data["val"])

    # Predict on Test
    print("Generating MLP predictions...")
    mlp_preds_test = mlp_model.predict_proba(feature_data["test"])

    # Verify Predictions
    assert len(mlp_preds_test) == feature_data["test"]["rf_features"].shape[0]
    assert np.all((mlp_preds_test >= 0) & (mlp_preds_test <= 1))
    print(f"MLP Test Predictions generated. Shape: {mlp_preds_test.shape}")

    # 5. Ensemble and Submission
    print("\n--- Step 4: Ensembling and Submission ---")

    # Weighted Average
    w_rf = Config.ENSEMBLE_WEIGHTS["rf"]
    w_mlp = Config.ENSEMBLE_WEIGHTS["mlp"]

    final_preds = (w_rf * rf_preds_test) + (w_mlp * mlp_preds_test)

    # Load Test Metadata to get request_ids
    # We use load_data just to get the dataframe for IDs
    _, _, df_test = load_data(
        load_cached_data=True
    )  # Use cache this time as it was generated in step 1

    assert len(df_test) == len(
        final_preds
    ), "Mismatch between test ID count and prediction count"

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {"request_id": df_test["request_id"], "requester_received_pizza": final_preds}
    )

    # Save
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to: {submission_path}")
    print("Head of submission:")
    print(submission_df.head())

    print("\nDemo Execution Completed Successfully.")


if __name__ == "__main__":
    run_demo()
