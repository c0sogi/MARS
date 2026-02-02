import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import sys

# Import provided library modules
import library.config as config
import library.train as train
import library.data as data
import library.utils as utils
import library.model as model_lib


def main():
    # 1. Train the model
    # run_training handles the full training loop, early stopping, and returns the best model.
    print("Starting training process...")
    trained_model, test_loader = train.run_training()

    # 2. Validation & Failure Analysis Setup
    # We need the validation loader to compute the final metric and analyze failures.
    # Using load_cached_data=True ensures we don't re-process the data.
    print("Loading validation data for evaluation...")
    _, val_loader, _, _ = data.get_dataloaders(load_cached_data=True)

    device = utils.get_device()
    trained_model.eval()

    all_preds = []
    all_targets = []
    all_X_cont = []

    # Inference on Validation Set
    print("Running inference on validation set...")
    with torch.no_grad():
        for X_cont, X_bin, y in val_loader:
            X_cont = X_cont.to(device)
            X_bin = X_bin.to(device)
            y = y.to(device)

            outputs = trained_model(X_cont, X_bin)
            _, preds = torch.max(outputs, 1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            # Store continuous features for failure analysis
            all_X_cont.append(X_cont.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_X_cont = np.concatenate(all_X_cont, axis=0)

    # 3. Compute and Print Final Metric
    accuracy = (all_preds == all_targets).mean()
    print(f"Final Validation Metric: {accuracy}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")
    errors = (all_preds != all_targets).astype(int)

    # Retrieve feature names to make the analysis interpretable
    # We load a tiny chunk of data and apply the same engineering to get column names
    try:
        df_head = pd.read_parquet(config.TRAIN_PATH).head(10)
        # Apply feature engineering to get the generated columns
        df_head = data._apply_feature_engineering(df_head)

        # Identify continuous columns using the same logic as data.py
        bin_cols = [
            c
            for c in df_head.columns
            if c.startswith("Wilderness_Area") or c.startswith("Soil_Type")
        ]
        exclude = ["Id", "Cover_Type"] + bin_cols
        cont_cols = [c for c in df_head.columns if c not in exclude]

        correlations = []
        for i, col_name in enumerate(cont_cols):
            if i < all_X_cont.shape[1]:
                feat_vals = all_X_cont[:, i]
                # Calculate correlation if variance is non-zero
                if np.std(feat_vals) > 1e-9:
                    corr, _ = pearsonr(feat_vals, errors)
                    correlations.append((col_name, corr))
                else:
                    correlations.append((col_name, 0.0))

        # Sort by absolute correlation magnitude
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print("Top 5 Continuous Features Correlated with Error:")
        for name, corr in correlations[:5]:
            print(f"  {name}: {corr:.4f}")

    except Exception as e:
        print(
            f"Warning: Could not perform detailed feature naming in failure analysis: {e}"
        )

    # 5. Submission Logic
    threshold = 0.9626291666666666
    if accuracy > threshold:
        print(f"\nValidation metric {accuracy} > {threshold}. Generating submission...")

        test_preds = []
        test_ids = []

        trained_model.eval()
        with torch.no_grad():
            for X_cont, X_bin, ids in test_loader:
                X_cont = X_cont.to(device)
                X_bin = X_bin.to(device)

                outputs = trained_model(X_cont, X_bin)
                _, preds = torch.max(outputs, 1)

                test_preds.append(preds.cpu().numpy())
                test_ids.append(ids.numpy())

        test_preds = np.concatenate(test_preds)
        test_ids = np.concatenate(test_ids)

        # Shift predictions from 0-6 back to 1-7 class labels
        test_preds_shifted = test_preds + 1

        utils.save_submission(test_preds_shifted, test_ids)
    else:
        print(f"\nValidation metric {accuracy} <= {threshold}. Skipping submission.")


if __name__ == "__main__":
    main()
