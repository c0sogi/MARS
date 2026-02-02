import sys
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Import from provided library
from library.config import Config
from library.utils import set_seed, save_submission, load_metadata_splits
from library.rf_manager import RFManager
from library.mlp_manager import MLPManager
from library.dataset_factory import get_dataloaders
from library.neural_architecture import OrthogonalSkipGatedMLP


def main():
    # 1. Setup
    set_seed(Config.RANDOM_SEED)
    print("Starting execution...")

    # 2. Train Random Forest (Stream A)
    rf_manager = RFManager()
    # This trains the model in memory and prints its internal validation score
    rf_manager.train(load_cached_data=True)

    # 3. Train MLP (Stream B)
    mlp_manager = MLPManager()
    # This trains the model, saves the best weights to disk, and prints progress
    mlp_manager.train(load_cached_data=True)

    # 4. Ensemble Validation
    print("\n--- Running Ensemble Validation ---")

    # Load metadata to get ground truth
    train_df, val_df, test_df = load_metadata_splits()
    y_val = val_df[Config.TARGET_COL].values.astype(int)

    # A. Get RF Validation Predictions
    # We access the protected method to get the exact features used during training
    X_val_rf = rf_manager._get_assembled_features(
        val_df, "val", Config.TRAIN_JSON_PATH, load_cached_data=True
    )
    # Predict probability of class 1
    rf_val_probs = rf_manager.model.predict_proba(X_val_rf)[:, 1]

    # B. Get MLP Validation Predictions
    # We need to manually run inference on the validation set using the best saved model
    # Get dataloaders (we only need val_loader here)
    _, val_loader, _, feature_dims = get_dataloaders(
        batch_size=Config.MLP_BATCH_SIZE, load_cached_data=True, num_workers=0
    )

    # Re-initialize model structure
    metadata_dim = feature_dims["metadata_dim"]
    mlp_model = OrthogonalSkipGatedMLP(metadata_dim)
    mlp_model.to(Config.DEVICE)

    # Load best weights
    if not os.path.exists(mlp_manager.model_path):
        raise FileNotFoundError("MLP model weights not found. Training likely failed.")

    mlp_model.load_state_dict(
        torch.load(mlp_manager.model_path, map_location=Config.DEVICE)
    )
    mlp_model.eval()

    mlp_val_probs = []
    with torch.no_grad():
        for batch_inputs, _ in val_loader:
            # Move inputs to device
            batch_inputs_dev = {k: v.to(Config.DEVICE) for k, v in batch_inputs.items()}

            # Forward pass
            logits = mlp_model(batch_inputs_dev)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

            # Handle scalar vs array output
            if np.ndim(probs) == 0:
                mlp_val_probs.append(float(probs))
            else:
                mlp_val_probs.extend(probs.tolist())

    mlp_val_probs = np.array(mlp_val_probs)

    # C. Compute Ensemble Score
    # Simple weighted average
    ensemble_val_probs = (Config.ENSEMBLE_WEIGHT_RF * rf_val_probs) + (
        Config.ENSEMBLE_WEIGHT_MLP * mlp_val_probs
    )

    final_auc = roc_auc_score(y_val, ensemble_val_probs)
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude
    errors = np.abs(y_val - ensemble_val_probs)

    # Identify numerical columns for correlation analysis
    numeric_cols = val_df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove target and potential IDs/timestamps that aren't useful features
    exclude_cols = [
        Config.TARGET_COL,
        "unix_timestamp_of_request",
        "unix_timestamp_of_request_utc",
    ]
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    correlations = {}
    for col in numeric_cols:
        # Fill NaNs with 0 for correlation calculation
        feat_values = val_df[col].fillna(0).values

        # Calculate correlation if variance exists
        if np.std(feat_values) > 0 and np.std(errors) > 0:
            corr = np.corrcoef(feat_values, errors)[0, 1]
            if not np.isnan(corr):
                correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for name, val in sorted_corr[:5]:
        print(f"{name:<50}: {val:.4f}")

    # 6. Submission
    threshold = 0.7135451153926904

    if final_auc > threshold:
        print(
            f"\nValidation metric {final_auc} > {threshold}. Generating submission..."
        )

        # Get Test Predictions from Managers
        rf_test_probs = rf_manager.predict_test(load_cached_data=True)
        mlp_test_probs = mlp_manager.predict_test(load_cached_data=True)

        # Ensemble
        ensemble_test_probs = (Config.ENSEMBLE_WEIGHT_RF * rf_test_probs) + (
            Config.ENSEMBLE_WEIGHT_MLP * mlp_test_probs
        )

        # Save
        test_ids = test_df[Config.ID_COL].values
        save_submission(test_ids, ensemble_test_probs)

    else:
        print(f"\nValidation metric {final_auc} <= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
