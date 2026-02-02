import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import provided library functions
from library.utils import set_seed, get_device
from library.dataset import load_dataset
from library.model import MGMTNet
from library.train import run_training


def main():
    # 1. Setup
    set_seed(42)
    device = get_device()
    working_dir = "./working/idea_11"
    best_model_path = os.path.join(working_dir, "best_model.pth")

    print("Starting pipeline...")

    # 2. Training
    # We use the full dataset (small size) but limit epochs to ensure quick execution.
    # The run_training function handles the training loop and saves the best model.
    print("Initiating training...")
    _ = run_training(
        num_epochs=20,
        batch_size=16,
        patience=5,
        load_cached_data=True,
        learning_rate=1e-4,
        save_dir=working_dir,
    )

    # 3. Validation Assessment
    print("Performing validation assessment...")

    # Load the best model
    model = MGMTNet().to(device)
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Error: Best model checkpoint not found.")
        return

    model.eval()

    # Load validation dataset
    val_dataset = load_dataset("val", load_cached_data=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    # Run inference
    val_targets = []
    val_probs = []
    val_ids = []  # To map back to metadata

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            # Targets are needed for metric calculation

            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            val_targets.extend(targets.numpy().flatten())
            val_probs.extend(probs)
            # Note: BraTSDataset returns (X, y) if y exists, so we rely on dataset.ids for ID mapping

    val_targets = np.array(val_targets)
    val_probs = np.array(val_probs)

    # Calculate Metric
    final_val_auc = roc_auc_score(val_targets, val_probs)
    print(f"Final Validation Metric: {final_val_auc}")

    # 4. Failure Analysis
    print("Running failure analysis...")
    errors = np.abs(val_targets - val_probs)

    # Load metadata to get features
    val_meta_df = pd.read_parquet("./metadata/val.parquet")

    # Align metadata with predictions using IDs
    # The dataset loads data in the order of the dataframe rows it processed.
    # We can verify alignment or just merge.
    val_ids = val_dataset.ids

    # Create a DataFrame for analysis
    analysis_df = pd.DataFrame({"BraTS21ID": val_ids, "error": errors})

    # Merge with metadata to get path lists
    analysis_df = analysis_df.merge(val_meta_df, on="BraTS21ID", how="left")

    # Extract meta-features (slice counts)
    modalities = ["flair", "t1w", "t1wce", "t2w"]
    correlations = {}

    print("Correlation between Model Error and Input Features:")
    for mod in modalities:
        col_name = f"{mod}_paths"
        if col_name in analysis_df.columns:
            # Calculate slice count
            counts = analysis_df[col_name].apply(
                lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
            )

            # Calculate correlation
            if len(counts) > 1 and np.std(counts) > 0:
                corr = np.corrcoef(analysis_df["error"], counts)[0, 1]
                correlations[f"{mod}_count"] = corr
                print(f"Feature: {mod}_slice_count | Correlation with Error: {corr}")
            else:
                print(
                    f"Feature: {mod}_slice_count | Correlation: N/A (Constant or Empty)"
                )

    # 5. Submission
    THRESHOLD = 0.6978181818181817

    if final_val_auc > THRESHOLD:
        print(
            f"Validation AUC ({final_val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        test_dataset = load_dataset("test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset, batch_size=16, shuffle=False, num_workers=2
        )

        test_ids = []
        test_probs = []

        with torch.no_grad():
            for inputs, ids in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()

                test_ids.extend(ids)
                test_probs.extend(probs)

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": test_probs})

        # Save
        os.makedirs("./submission", exist_ok=True)
        submission_path = "./submission/submission.csv"
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"Validation AUC ({final_val_auc}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
