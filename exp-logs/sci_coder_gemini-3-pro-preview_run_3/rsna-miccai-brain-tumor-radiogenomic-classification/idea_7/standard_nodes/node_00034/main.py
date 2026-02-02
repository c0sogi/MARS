import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from library import config, data_loader, model as model_lib, train_eval
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # Set seed for reproducibility
    train_eval.set_seed(config.SEED)

    print("Initializing pipeline...")

    # 1. Train the model
    # We use the full dataset as it is small (approx 400 training samples)
    # and fits within the time/compute constraints easily.
    # This function returns the best AUC achieved during training epochs.
    print("Starting training...")
    _ = train_eval.run_training(load_cached_data=True)

    # 2. Validation Inference & Metric Calculation
    print("Performing validation inference...")
    device = config.DEVICE

    # Load validation data
    # We access the val_loader directly to ensure we use the exact same split
    _, val_loader, _ = data_loader.get_dataloaders(load_cached_data=True)

    # Load the best model saved during training
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        print("Error: Best model file not found.")
        return

    model = model_lib.MGMTNet()
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    # Inference loop (no grad for speed/memory optimization)
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            # targets are already tensors in loader

            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(targets.numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate Final Metric
    final_metric = roc_auc_score(all_targets, all_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 3. Failure Analysis
    print("Performing failure analysis...")
    # Calculate error magnitude
    errors = np.abs(all_targets - all_preds)

    # Load metadata to get features for correlation
    val_df = pd.read_parquet(config.VAL_META_PATH)

    # Ensure dataframe length matches predictions
    if len(val_df) != len(errors):
        print(
            f"Warning: Mismatch in validation set size. DF: {len(val_df)}, Preds: {len(errors)}"
        )
        min_len = min(len(val_df), len(errors))
        val_df = val_df.iloc[:min_len]
        errors = errors[:min_len]

    # Extract features: Slice counts per modality
    # These are good proxies for "input features" regarding data quality/quantity
    features = {}
    for mod in config.MODALITIES:  # ["FLAIR", "T1w", "T1wCE", "T2w"]
        # Metadata columns are lowercase e.g. "flair_paths"
        col_name = f"{mod.lower()}_paths"
        if col_name in val_df.columns:
            features[f"{mod}_count"] = val_df[col_name].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )

    if features:
        feature_df = pd.DataFrame(features)
        feature_df["error"] = errors

        # Calculate correlation
        correlations = feature_df.corr()["error"].drop("error")
        print("Correlation between Error and Features:")
        print(correlations)
    else:
        print("Could not extract features for failure analysis.")

    # 4. Submission
    # Threshold from task description
    THRESHOLD = 0.6978181818181817

    if final_metric > THRESHOLD:
        print("Validation metric meets threshold. Generating submission...")
        train_eval.generate_submission(load_cached_data=True)
    else:
        print(
            f"Validation metric {final_metric} does not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
