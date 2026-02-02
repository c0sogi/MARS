import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed
from library.data import prepare_datasets
from library.train import run_training
from library.model import DualStreamSiameseNet


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Train Model
    # run_training handles data loading (with caching), model init, training loop,
    # and returns the model with the best weights loaded.
    model = run_training(load_cached_data=True)

    # 3. Validation & Metric Calculation
    # We reload datasets to ensure we have access to the validation set object
    _, val_dataset, test_dataset = prepare_datasets(load_cached_data=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    model.eval()
    val_preds = []
    val_labels = []

    # Inference on Validation Set
    with torch.no_grad():
        for x_even, x_odd, labels in val_loader:
            x_even = x_even.to(device)
            x_odd = x_odd.to(device)

            outputs = model(x_even, x_odd)
            probs = torch.sigmoid(outputs).cpu().numpy()

            val_preds.extend(probs.flatten())
            val_labels.extend(labels.numpy().flatten())

    val_preds = np.array(val_preds)
    val_labels = np.array(val_labels)

    # Calculate and Print Metric
    final_metric = roc_auc_score(val_labels, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    errors = np.abs(val_labels - val_preds)

    # Load metadata to correlate errors with input properties (e.g., slice counts)
    val_meta_path = os.path.join(Config.METADATA_DIR, "val.parquet")
    if os.path.exists(val_meta_path):
        val_df = pd.read_parquet(val_meta_path)

        # Align metadata with validation predictions using BraTS21ID
        # val_dataset.ids contains the IDs in the order of the dataset
        val_ids = val_dataset.ids
        val_df = val_df.set_index("BraTS21ID").loc[val_ids].reset_index()

        modalities = ["flair", "t1w", "t1wce", "t2w"]
        print("Correlation between Error Magnitude and Slice Counts per Modality:")

        for mod in modalities:
            col_name = f"{mod}_paths"
            # Calculate slice count for each patient
            slice_counts = (
                val_df[col_name].apply(lambda x: len(x) if x is not None else 0).values
            )

            # Compute correlation if variance exists
            if np.std(slice_counts) > 0 and np.std(errors) > 0:
                corr, _ = pearsonr(errors, slice_counts)
                print(f"  {mod} slice count: {corr:.4f}")
            else:
                print(f"  {mod} slice count: N/A (No variance)")
    else:
        print("Validation metadata not found, skipping detailed failure analysis.")

    # 5. Submission Generation
    THRESHOLD = 0.6978181818181817

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_preds = []
        test_ids = []

        with torch.no_grad():
            for x_even, x_odd, ids in test_loader:
                x_even = x_even.to(device)
                x_odd = x_odd.to(device)

                outputs = model(x_even, x_odd)
                probs = torch.sigmoid(outputs).cpu().numpy()

                test_preds.extend(probs.flatten())
                test_ids.extend(ids)

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": test_preds})

        # Save to file
        os.makedirs("submission", exist_ok=True)
        submission_path = "submission/submission.csv"
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
