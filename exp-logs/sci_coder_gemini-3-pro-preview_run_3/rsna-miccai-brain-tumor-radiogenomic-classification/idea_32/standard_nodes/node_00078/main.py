import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from library.config import WORKING_DIR, SUBMISSION_PATH, DEVICE
from library.utils import seed_everything
from library.data import load_processed_data, BraTSDataset
from library.model import HRLNNet
from library.train import train_model


def main():
    # 1. Setup
    seed_everything()
    print("Initializing Pipeline...")

    # 2. Train Model
    # Running for 10 epochs to ensure a fast baseline execution while allowing convergence.
    # load_cached_data=True ensures we use preprocessed .npy files if available.
    best_model_path = train_model(load_cached_data=True, num_epochs=10)

    # 3. Validation Inference
    print("\nRunning Validation Inference...")
    # Load validation data
    X_val, y_val, ids_val = load_processed_data("val", load_cached_data=True)
    val_dataset = BraTSDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

    # Load Model
    model = HRLNNet().to(DEVICE)
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(DEVICE)
            # No need to move targets to device for inference-only metric calc,
            # but consistent with training loop

            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_preds.extend(probs)
            all_targets.extend(targets.numpy())

    # 4. Metric Reporting
    val_auc = roc_auc_score(all_targets, all_preds)
    print(f"Final Validation Metric: {val_auc}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    errors = np.abs(np.array(all_targets) - np.array(all_preds))

    # Load metadata to extract input features (slice counts)
    val_meta_path = "./metadata/val.parquet"
    if os.path.exists(val_meta_path):
        val_df = pd.read_parquet(val_meta_path)

        # Ensure alignment: The data loader loads in order of the parquet file
        # We extract slice counts for each modality
        feature_correlations = {}
        modalities = ["flair", "t1w", "t1wce", "t2w"]

        print("Correlation between Error Magnitude and Input Features (Slice Counts):")
        for mod in modalities:
            # Calculate slice count for each subject
            counts = (
                val_df[f"{mod}_paths"]
                .apply(lambda x: len(x) if x is not None else 0)
                .values
            )

            # Calculate correlation if lengths match
            if len(counts) == len(errors):
                corr = np.corrcoef(errors, counts)[0, 1]
                print(f"  {mod}_count: {corr:.6f}")
            else:
                print(
                    f"  {mod}_count: Size mismatch (Data: {len(errors)}, Meta: {len(counts)})"
                )
    else:
        print("Validation metadata not found, skipping detailed feature correlation.")

    # 6. Submission Logic
    THRESHOLD = 0.6978181818181817

    if val_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({val_auc:.6f}) exceeds threshold ({THRESHOLD:.6f}). Generating submission..."
        )

        # Load Test Data
        X_test, _, ids_test = load_processed_data("test", load_cached_data=True)
        test_dataset = BraTSDataset(X_test, None)
        test_loader = DataLoader(
            test_dataset, batch_size=8, shuffle=False, num_workers=2
        )

        test_preds = []

        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(DEVICE)
                logits = model(inputs)
                probs = torch.sigmoid(logits).cpu().numpy()
                test_preds.extend(probs)

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"BraTS21ID": ids_test, "MGMT_value": test_preds})

        # Ensure output directory exists
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation AUC ({val_auc:.6f}) does not exceed threshold ({THRESHOLD:.6f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
