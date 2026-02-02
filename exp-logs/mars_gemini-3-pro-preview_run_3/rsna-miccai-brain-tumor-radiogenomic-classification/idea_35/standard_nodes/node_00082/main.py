import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import provided library functions
from library.utils import seed_everything, get_device, load_data
from library.data_loader import DualStreamDataset
from library.model import DSSVNet
from library.train import run_training
from library.predict import generate_submission


def main():
    # ==========================================
    # Configuration
    # ==========================================
    SEED = 42
    BATCH_SIZE = 16
    EPOCHS = 15
    LR = 1e-4
    INPUT_DIR = "./input"
    CACHE_DIR = "./working/idea_35/"
    METADATA_DIR = "./metadata"
    SUBMISSION_PATH = "./submission/submission.csv"
    AUC_THRESHOLD = 0.6978181818181817

    # Ensure reproducibility
    seed_everything(SEED)
    device = get_device()

    print(f"Execution started on device: {device}")

    # ==========================================
    # 1. Training
    # ==========================================
    # We use the full dataset (limit_size=None) as the dataset is small (~400 samples)
    # and fits easily within the time limit on an A100 GPU.
    print("Starting training pipeline...")
    best_model_path = run_training(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        input_dir=INPUT_DIR,
        cache_dir=CACHE_DIR,
        limit_size=None,
        seed=SEED,
    )

    # ==========================================
    # 2. Validation & Failure Analysis
    # ==========================================
    print("Starting validation and failure analysis...")

    # Load Validation Data
    # We explicitly load it here to have access to targets and inputs for analysis
    X_val, y_val, ids_val = load_data(
        split="val", load_cached_data=True, cache_dir=CACHE_DIR, input_dir=INPUT_DIR
    )

    # Load the best model
    model = DSSVNet(pretrained=False)
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    # Create DataLoader for inference
    val_dataset = DualStreamDataset(X_val, y_val, ids_val)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Run Inference
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for (even_stream, odd_stream), targets in val_loader:
            even_stream = even_stream.to(device)
            odd_stream = odd_stream.to(device)

            # Forward pass
            logits = model(even_stream, odd_stream)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(targets.numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute Metric
    final_auc = roc_auc_score(all_targets, all_preds)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    # We correlate the absolute error with the number of slices per modality
    # to see if data quantity affects performance.
    errors = np.abs(all_preds - all_targets)

    # Load metadata to get slice counts
    val_meta_path = os.path.join(METADATA_DIR, "val.parquet")
    val_df = pd.read_parquet(val_meta_path)

    # Ensure alignment: The load_data function processes rows in order of the dataframe.
    # We double check lengths match.
    if len(val_df) != len(errors):
        print(
            "Warning: Metadata length mismatch with validation set size. Skipping detailed correlation."
        )
    else:
        print("\nFailure Analysis (Correlation of Error with Features):")
        modalities = ["flair", "t1w", "t1wce", "t2w"]

        # Calculate slice counts for each patient
        for mod in modalities:
            col_name = f"{mod}_paths"
            # Count items in the list column
            counts = val_df[col_name].apply(
                lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
            )

            # Calculate Pearson correlation
            corr, _ = pearsonr(errors, counts)
            print(f"Error vs {mod}_slice_count: {corr:.4f}")

        # Also correlate with target class
        corr_target, _ = pearsonr(errors, all_targets)
        print(f"Error vs Target Class: {corr_target:.4f}")

    # ==========================================
    # 3. Submission
    # ==========================================
    if final_auc > AUC_THRESHOLD:
        print(
            f"\nValidation metric {final_auc} exceeds threshold {AUC_THRESHOLD}. Generating submission..."
        )
        generate_submission(
            model_path=best_model_path,
            output_file=SUBMISSION_PATH,
            input_dir=INPUT_DIR,
            cache_dir=CACHE_DIR,
            batch_size=BATCH_SIZE,
            limit_size=None,
            seed=SEED,
        )
    else:
        print(
            f"\nValidation metric {final_auc} does not exceed threshold {AUC_THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
