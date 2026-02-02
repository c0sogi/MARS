import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, DataLoader

# Import provided library modules
from library import config
from library import utils
from library import data_loader
from library import model
from library import train
from library import predict


def perform_failure_analysis(val_ids, val_probs, val_targets):
    """
    Analyzes the correlation between prediction error and input meta-features.
    """
    print("\nPerforming Failure Analysis...")

    # Load metadata to get features like slice counts
    val_df = pd.read_parquet(config.VAL_META_PATH)

    # Create a DataFrame for predictions
    pred_df = pd.DataFrame(
        {"BraTS21ID": val_ids, "prob": val_probs, "target": val_targets}
    )

    # Merge with metadata to ensure alignment
    # BraTS21ID in metadata is string, ensure consistency
    pred_df["BraTS21ID"] = pred_df["BraTS21ID"].astype(str)
    val_df["BraTS21ID"] = val_df["BraTS21ID"].astype(str)

    merged_df = pd.merge(val_df, pred_df, on="BraTS21ID", how="inner")

    # Calculate Error Magnitude
    merged_df["error"] = np.abs(merged_df["target"] - merged_df["prob"])

    # Extract Meta-Features (Slice Counts)
    modalities = ["flair", "t1w", "t1wce", "t2w"]
    correlations = {}

    print("Correlation between Error Magnitude and Slice Counts:")
    for mod in modalities:
        col_name = f"{mod}_paths"
        # Calculate count of slices
        merged_df[f"{mod}_count"] = merged_df[col_name].apply(
            lambda x: len(x) if x is not None else 0
        )

        # Calculate correlation
        if merged_df[f"{mod}_count"].std() > 0:
            corr, _ = pearsonr(merged_df["error"], merged_df[f"{mod}_count"])
            correlations[f"{mod}_count"] = corr
            print(f" - {mod}_count: {corr:.4f}")
        else:
            print(f" - {mod}_count: NaN (No variance)")


def main():
    # 1. Setup
    utils.seed_everything(config.SEED)
    device = utils.get_device()

    # 2. Train the Model
    # We use max_samples=None to use the full dataset for best performance
    # The dataset is small enough (~500) to run quickly.
    train.run_training(max_samples=None)

    # 3. Load Best Model for Validation
    print("\nLoading best model for final validation assessment...")
    net = model.MGMTNet()
    net = net.to(device)

    if not os.path.exists(config.MODEL_SAVE_PATH):
        print("Error: Model file not found.")
        return

    state_dict = torch.load(config.MODEL_SAVE_PATH, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()

    # 4. Load Validation Data
    # We load directly using get_data_for_split to get arrays
    val_X, val_y, val_ids = data_loader.get_data_for_split(
        "val", config.VAL_META_PATH, load_cached_data=True, max_samples=None
    )

    val_dataset = TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_y))
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 5. Run Inference on Validation
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            logits = net(inputs)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    all_probs = np.concatenate(all_probs).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    # 6. Calculate and Print Metric
    final_auc = roc_auc_score(all_targets, all_probs)
    print(f"Final Validation Metric: {final_auc}")

    # 7. Failure Analysis
    perform_failure_analysis(val_ids, all_probs, all_targets)

    # 8. Conditional Submission
    # Threshold from instructions
    THRESHOLD = 0.6978181818181817

    if final_auc > THRESHOLD:
        print(
            f"\nValidation metric ({final_auc}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        predict.run_prediction(load_cached_data=True, max_samples=None)
    else:
        print(
            f"\nValidation metric ({final_auc}) does not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
