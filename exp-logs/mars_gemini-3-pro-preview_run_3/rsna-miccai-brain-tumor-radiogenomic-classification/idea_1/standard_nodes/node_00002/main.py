import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import from provided libraries
from library.config import Config
from library.utils import set_seed
from library.dataset import MGMTDataset
from library.model import MGMTNet
from library.train import run_training
from library.predict import generate_submission


def perform_failure_analysis(model, device):
    """
    Analyzes model errors on the validation set by correlating error magnitude
    with input meta-features (slice counts).
    """
    print("\n" + "=" * 40)
    print(" FAILURE ANALYSIS")
    print("=" * 40)

    # Load validation metadata
    val_df = pd.read_parquet(Config.VAL_METADATA)

    # Create dataset and loader
    val_dataset = MGMTDataset(val_df, split_name="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Run Inference
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate Error Magnitude
    errors = np.abs(all_preds - all_targets)

    # Extract Meta-Features (Slice Counts)
    # The dataset/loader might shuffle or process, but since shuffle=False
    # and we read the parquet directly, the order should align.
    # We'll re-verify alignment is not needed as shuffle=False.

    meta_features = {}
    modalities = ["flair", "t1w", "t1wce", "t2w"]

    for mod in modalities:
        col_name = f"{mod}_paths"
        # Calculate number of slices for each patient
        counts = val_df[col_name].apply(lambda x: len(x) if x is not None else 0)
        meta_features[f"{mod}_count"] = counts.values

    # Calculate Correlations
    print("Correlation between Error Magnitude and Slice Counts:")
    for feature_name, values in meta_features.items():
        if len(values) != len(errors):
            print(f"Warning: Shape mismatch for {feature_name}. Skipping.")
            continue

        # Compute Point-Biserial or Pearson correlation
        # Since slice counts are continuous-ish and error is continuous
        corr = np.corrcoef(values, errors)[0, 1]
        print(f" - {feature_name}: {corr:.4f}")

    # Calculate Final Metric again to ensure it matches the requirement exactly
    try:
        final_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        final_auc = 0.5

    return final_auc


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Training
    print("\nStarting Training Pipeline...")
    run_training(
        debug=False,
        epochs=Config.NUM_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        patience=Config.EARLY_STOPPING_PATIENCE,
        num_workers=Config.NUM_WORKERS,
        device_name=Config.DEVICE,
        save_path=Config.MODEL_PATH,
    )

    # 3. Validation & Failure Analysis
    # Load the best model
    model = MGMTNet().to(device)
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print(f"\nLoaded best model from {Config.MODEL_PATH}")
    else:
        print("\nWarning: Model checkpoint not found! Using random weights.")

    # Perform analysis and get final metric
    final_metric = perform_failure_analysis(model, device)

    # Print required metric format
    print(f"Final Validation Metric: {final_metric}")

    # 4. Submission
    print("\n" + "=" * 40)
    print(" GENERATING SUBMISSION")
    print("=" * 40)
    generate_submission(
        model_path=Config.MODEL_PATH,
        output_path=Config.SUBMISSION_PATH,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        device_name=Config.DEVICE,
    )
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
