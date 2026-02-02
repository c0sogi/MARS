import os
import sys
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Ensure library imports work correctly
sys.path.append(os.getcwd())

from library.config import Config, setup_reproducibility
from library.train import run_training
from library.predict import predict_submission
from library.model import AsymmetricEfficientNet
from library.data import get_dataloaders


def perform_failure_analysis(model, val_loader, device):
    """
    Runs inference on validation set, calculates errors, and correlates with metadata.
    """
    print("\n--- Performing Failure Analysis ---")
    model.eval()

    all_preds = []
    all_targets = []

    # Run Inference
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(labels.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate Metric
    try:
        val_auc = roc_auc_score(all_targets, all_preds)
    except Exception:
        val_auc = 0.5

    print(f"Final Validation Metric: {val_auc}")

    # Load Validation Metadata to get features for correlation
    val_df = pd.read_csv(Config.METADATA_VAL)

    # Verify alignment
    if len(val_df) != len(all_preds):
        print(
            f"Warning: Metadata length ({len(val_df)}) != Predictions length ({len(all_preds)}). Skipping detailed correlation analysis."
        )
        return val_auc

    # Calculate Error Magnitude
    errors = np.abs(all_targets - all_preds)
    val_df["error"] = errors

    # Extract Metadata Features for Correlation
    modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]
    feature_cols = []

    print("Extracting metadata features for correlation...")
    for mod in modalities:
        col_slices = f"{mod}_slices"
        col_size = f"{mod}_avg_size"
        feature_cols.extend([col_slices, col_size])

        slices_list = []
        sizes_list = []

        for _, row in val_df.iterrows():
            path = os.path.join(Config.INPUT_DIR, row[f"path_{mod}"])
            if os.path.exists(path):
                files = os.listdir(path)
                n_slices = len(files)
                slices_list.append(n_slices)

                if n_slices > 0:
                    # Use size of the first file as a proxy for resolution/quality
                    try:
                        f_path = os.path.join(path, files[0])
                        sizes_list.append(os.path.getsize(f_path))
                    except:
                        sizes_list.append(0)
                else:
                    sizes_list.append(0)
            else:
                slices_list.append(0)
                sizes_list.append(0)

        val_df[col_slices] = slices_list
        val_df[col_size] = sizes_list

    # Calculate Correlations
    print("\nCorrelation between Error Magnitude and Metadata Features:")
    correlations = {}
    for col in feature_cols:
        if val_df[col].std() > 0:  # Avoid constant columns
            corr, _ = pearsonr(val_df["error"], val_df[col])
            correlations[col] = corr
            print(f"  {col}: {corr:.4f}")
        else:
            print(f"  {col}: NaN (Constant)")

    return val_auc


def main():
    # 1. Setup
    setup_reproducibility(Config.SEED)

    # 2. Run Training
    # This handles data caching, model training, and saving the best model.
    print(">>> Starting Training Pipeline")
    run_training(load_cached=True)

    # 3. Load Best Model for Analysis
    print("\n>>> Loading Best Model for Analysis")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AsymmetricEfficientNet()
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Error: Best model weights not found.")
        return

    model.to(device)

    # 4. Get Validation Loader
    # We ignore train/test loaders here, just need val
    _, val_loader, _, _ = get_dataloaders(load_cached=True)

    # 5. Perform Failure Analysis & Get Final Metric
    final_metric = perform_failure_analysis(model, val_loader, device)

    # 6. Submission Logic
    threshold = 0.6321818181818182
    print(f"\nThreshold: {threshold}")
    print(f"Achieved:  {final_metric}")

    if final_metric > threshold:
        print(">>> Metric condition met. Generating submission...")
        predict_submission(load_cached=True)
    else:
        print(">>> Metric condition NOT met. Skipping submission generation.")


if __name__ == "__main__":
    main()
