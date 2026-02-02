import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Ensure library modules can be imported
sys.path.append(os.getcwd())

import library.config as config
from library.data_utils import load_data
from library.dataset import ManufacturingDataset
from library.model import MRPFEModel
from library.train_eval import run_training, generate_submission, set_seed


def main():
    # 1. Setup
    set_seed(config.SEED)

    # 2. Train the model
    # We use the full dataset (nrows=None) and default config (50 epochs)
    # to ensure we hit the high performance threshold.
    print("Starting training pipeline...")
    best_model_path, vocab_sizes, num_continuous = run_training(nrows=None)

    # 3. Validation Assessment
    print("Performing validation assessment...")

    # Load validation data (utilizing cache generated during training)
    # We only need val_df here
    _, val_df, _, _ = load_data(load_cached_data=True)

    # Prepare Validation Loader
    val_dataset = ManufacturingDataset(val_df, is_test=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Best Model
    device = config.DEVICE
    model = MRPFEModel(vocab_sizes, num_continuous).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Inference on Validation Set
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in val_loader:
            continuous = batch["continuous"].to(device)
            categorical = batch["categorical"].to(device)
            targets = batch["target"].to(device)

            outputs = model(continuous, categorical)

            # Ensemble prediction: Arithmetic mean of probabilities from 5 streams
            probs_sum = 0
            for output in outputs:
                probs_sum += torch.sigmoid(output)

            avg_probs = probs_sum / len(outputs)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(avg_probs.cpu().numpy())

    all_targets = np.concatenate(all_targets).flatten()
    all_preds = np.concatenate(all_preds).flatten()

    # Calculate Metric
    val_auc = roc_auc_score(all_targets, all_preds)

    # REQUIRED OUTPUT: Final Validation Metric
    print(f"Final Validation Metric: {val_auc}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error magnitude
    errors = np.abs(all_targets - all_preds)

    # Create a dataframe for correlation analysis
    # We use the processed features in val_df.
    # Note: val_df contains scaled continuous features and encoded categorical features.
    # While correlation with encoded categoricals is rough, it provides the required signal.
    analysis_df = val_df[config.ALL_CONTINUOUS_FEATURES].copy()
    analysis_df["error_magnitude"] = errors

    # Compute correlation
    correlations = analysis_df.corr()["error_magnitude"].drop("error_magnitude")

    # Sort by absolute correlation to find strongest relationships (positive or negative)
    sorted_corrs = correlations.abs().sort_values(ascending=False)

    print("Top 10 Feature Correlations with Error Magnitude:")
    for feature_name in sorted_corrs.head(10).index:
        corr_val = correlations[feature_name]
        print(f"{feature_name}: {corr_val}")

    # 5. Submission
    THRESHOLD = 0.9975746465492954

    if val_auc > THRESHOLD:
        print(f"\nValidation metric {val_auc} exceeds threshold {THRESHOLD}.")
        print("Generating submission...")
        generate_submission(best_model_path, vocab_sizes, num_continuous)
    else:
        print(f"\nValidation metric {val_auc} does not meet threshold {THRESHOLD}.")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
