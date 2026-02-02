import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import pearsonr

# Import from library
from library.config import (
    WORKING_DIR,
    MODEL_SAVE_PATH,
    SCORED_LEN,
    TARGET_COLS,
    SCORED_TARGETS,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    SEED,
    VAL_CSV,
    SEQ_LEN,
)
from library.utils import MCRMSELoss, MCRMSE, parse_list_column
from library.model import HS_GFN
from library.data import get_loaders
from library.train import train_one_epoch, validate, get_scored_indices

# Constants
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
THRESHOLD = 0.47142532743789534


def set_seed(seed=SEED):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def perform_failure_analysis(model, val_loader, device, scored_indices):
    """
    Analyzes model performance on the validation set and correlates error with metadata.
    """
    print("\nPerforming Failure Analysis...")
    model.eval()

    all_errors = []
    all_ids = []

    # 1. Compute per-sample error
    with torch.no_grad():
        for x, partner_indices, targets in val_loader:
            x = x.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)  # (B, 5, L)

            # Inference (Pass 2)
            preds_1 = model(x, partner_indices, y_prev=None)
            y_feedback = preds_1.permute(0, 2, 1)
            preds_2 = model(x, partner_indices, y_prev=y_feedback)  # (B, L, 5)

            # Prepare targets
            targets_perm = targets.permute(0, 2, 1)  # (B, L, 5)

            # Select scored columns
            preds_scored = preds_2[:, :, scored_indices]
            targets_scored = targets_perm[:, :, scored_indices]

            # Mask for scored positions (0-67)
            # Calculate RMSE per sample
            # Error tensor: (B, L, C)
            squared_diff = (preds_scored - targets_scored) ** 2

            # Slice to scored length
            squared_diff = squared_diff[:, :SCORED_LEN, :]

            # Mean over Length and Channels for each sample
            mse_per_sample = squared_diff.mean(dim=(1, 2))
            rmse_per_sample = torch.sqrt(mse_per_sample)

            all_errors.extend(rmse_per_sample.cpu().numpy())

            # Get IDs for this batch
            # Note: The loader implementation in library/data.py doesn't yield IDs in __getitem__
            # but we can access them via the dataset if we iterate sequentially and don't shuffle.
            # However, RNADataset returns (x, p_idx, y).
            # We need to map these errors back to IDs.
            # Since val_loader shuffle=False, we can just accumulate errors and map to dataset.ids

    # 2. Load Metadata
    df_val = pd.read_csv(VAL_CSV)

    # Ensure alignment
    # The val_loader iterates over the dataset in order.
    # We double check lengths.
    if len(all_errors) != len(df_val):
        print(
            f"Warning: Mismatch in validation set size. Loader: {len(all_errors)}, CSV: {len(df_val)}"
        )
        # We will assume the loader corresponds to the CSV rows in order as shuffle=False

    df_val["model_error"] = all_errors[: len(df_val)]

    # 3. Calculate Correlations
    # Features to check
    features = ["signal_to_noise", "mean_reactivity", "seq_length"]
    # Add structure features if possible (e.g. count of open brackets)
    df_val["pct_paired"] = df_val["structure"].apply(
        lambda s: (s.count("(") + s.count(")")) / len(s)
    )
    features.append("pct_paired")

    print(f"{'Feature':<20} | {'Correlation with Error':<20} | {'P-value':<10}")
    print("-" * 60)

    for feat in features:
        if feat in df_val.columns:
            # Drop NaNs if any
            valid_df = df_val[[feat, "model_error"]].dropna()
            if len(valid_df) > 1:
                corr, p_val = pearsonr(valid_df[feat], valid_df["model_error"])
                print(f"{feat:<20} | {corr:.4f}               | {p_val:.4f}")
            else:
                print(f"{feat:<20} | Not enough data")
        else:
            print(f"{feat:<20} | Not found in metadata")

    print("-" * 60)


def generate_test_submission(model, test_loader, device):
    """Generates submission file for the test set."""
    print(f"Generating submission to {SUBMISSION_FILE}...")
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in test_loader:
            x, partner_indices, _ = batch
            x = x.to(device)
            partner_indices = partner_indices.to(device)

            # Pass 1
            preds_1 = model(x, partner_indices, y_prev=None)

            # Pass 2
            y_feedback = preds_1.permute(0, 2, 1)
            preds_2 = model(x, partner_indices, y_prev=y_feedback)  # (B, L, 5)

            all_preds.append(preds_2.cpu().numpy())

            # Get IDs from the batch indices?
            # The loader doesn't yield IDs. We must access dataset.ids
            # Since shuffle=False, we can just use the dataset.ids at the end.

    # Concatenate predictions
    preds_flat = np.concatenate(all_preds, axis=0)  # (N_samples, 107, 5)

    # Get IDs from dataset
    ids = test_loader.dataset.ids

    submission_data = []
    for i, sample_id in enumerate(ids):
        sample_preds = preds_flat[i]
        for seqpos in range(SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos]

            row = {"id_seqpos": row_id}
            for j, col_name in enumerate(TARGET_COLS):
                row[col_name] = row_values[j]
            submission_data.append(row)

    df_sub = pd.DataFrame(submission_data)
    cols = ["id_seqpos"] + TARGET_COLS
    df_sub = df_sub[cols]

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(SUBMISSION_FILE, index=False)
    print("Submission saved successfully.")


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 1. Load Data
    # load_cached_data=True to use preprocessed .npz files from ./working
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 2. Model Setup
    model = HS_GFN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = MCRMSELoss()
    scored_indices = get_scored_indices()

    # 3. Training Loop
    best_score = float("inf")
    patience = 10
    early_stop_count = 0

    print("Starting training...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scored_indices
        )
        val_score = validate(model, val_loader, device, scored_indices)

        # Scheduler step
        scheduler.step(val_score)

        # Checkpoint
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # 4. Final Validation
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

    final_metric = validate(model, val_loader, device, scored_indices)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    perform_failure_analysis(model, val_loader, device, scored_indices)

    # 6. Submission
    if final_metric < THRESHOLD:
        generate_test_submission(model, test_loader, device)
    else:
        print(
            f"Metric {final_metric} is not lower than threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
