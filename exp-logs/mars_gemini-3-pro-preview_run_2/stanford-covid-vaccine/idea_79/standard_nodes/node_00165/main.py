import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import scipy.stats as stats

from library.config import (
    SEED,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    LR_FACTOR,
    LR_PATIENCE,
    BEST_MODEL_PATH,
    SCORED_SEQ_LENGTH,
    SCORED_INDICES,
    ALL_TARGETS,
    SUBMISSION_PATH,
    VAL_CSV,
)
from library.utils import set_seed, mcrmse
from library.data import get_dataloaders
from library.model import RIS_DRN
from library.train import train_epoch, validate

# Limit epochs for fast baseline execution
FAST_EPOCHS = 5


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis by correlating sample-wise error with metadata features.
    """
    print("\nRunning Failure Analysis...")
    model.eval()

    # 1. Collect Predictions and Targets per ID
    results = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            partner_map = batch["partner_map"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            # Forward pass (use refined output logits_2)
            _, logits_2 = model(inputs, partner_map)

            # Move to CPU
            preds = logits_2.cpu().numpy()
            targs = targets.cpu().numpy()

            # Calculate error per sample
            for i in range(len(ids)):
                # Slice to scored region and scored columns
                p = preds[i, :SCORED_SEQ_LENGTH, :][:, SCORED_INDICES]
                t = targs[i, :SCORED_SEQ_LENGTH, :][:, SCORED_INDICES]

                # RMSE per column
                mse_per_col = np.mean((p - t) ** 2, axis=0)
                rmse_per_col = np.sqrt(mse_per_col)
                # MCRMSE for this sample
                sample_error = np.mean(rmse_per_col)

                results.append({"id": ids[i], "error": sample_error})

    results_df = pd.DataFrame(results)

    # 2. Load Metadata
    if not os.path.exists(VAL_CSV):
        print(
            f"Warning: Validation metadata not found at {VAL_CSV}. Skipping correlation analysis."
        )
        return

    meta_df = pd.read_csv(VAL_CSV)

    # 3. Merge
    analysis_df = pd.merge(results_df, meta_df, on="id", how="inner")

    # 4. Calculate Correlations
    # Features to check: signal_to_noise, SN_filter, mean_reactivity (if available)
    features = ["signal_to_noise", "SN_filter", "mean_reactivity"]

    print("Correlation between Model Error (MCRMSE) and Features:")
    for feat in features:
        if feat in analysis_df.columns:
            # Drop NaNs just in case
            valid_data = analysis_df[[feat, "error"]].dropna()
            if len(valid_data) > 1:
                corr, _ = stats.pearsonr(valid_data[feat], valid_data["error"])
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: Not enough data")
        else:
            print(f"  {feat}: Feature not found in metadata")


def generate_submission(model, test_loader, device):
    """
    Generates submission file for the test set.
    """
    print("\nGenerating Submission...")
    model.eval()

    submission_rows = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            partner_map = batch["partner_map"].to(device)
            ids = batch["id"]

            # Forward pass
            _, logits_2 = model(inputs, partner_map)
            preds = logits_2.cpu().numpy()  # (B, 107, 5)

            batch_size, seq_len, _ = preds.shape

            for i in range(batch_size):
                sample_id = ids[i]
                sample_preds = preds[i]  # (107, 5)

                for seqpos in range(seq_len):
                    # Row ID: id_seqpos
                    row_id = f"{sample_id}_{seqpos}"

                    # Values: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
                    # The model output order matches ALL_TARGETS in config
                    vals = sample_preds[seqpos]

                    row_dict = {
                        "id_seqpos": row_id,
                        "reactivity": vals[0],
                        "deg_Mg_pH10": vals[1],
                        "deg_pH10": vals[2],
                        "deg_Mg_50C": vals[3],
                        "deg_50C": vals[4],
                    }
                    submission_rows.append(row_dict)

    # Create DataFrame
    sub_df = pd.DataFrame(submission_rows)

    # Ensure column order
    cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    sub_df = sub_df[cols]

    # Save
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    sub_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


def main():
    # 1. Setup
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = RIS_DRN().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE
    )

    # 4. Training Loop (Fast Baseline)
    print(f"Starting training for {FAST_EPOCHS} epochs...")
    best_score = float("inf")

    for epoch in range(1, FAST_EPOCHS + 1):
        start_t = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler
        scheduler.step(val_score)

        # Save Best
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), BEST_MODEL_PATH)

        elapsed = time.time() - start_t
        print(
            f"Epoch {epoch} | Train Loss: {train_loss:.5f} | Val MCRMSE: {val_score:.5f} | Time: {elapsed:.1f}s"
        )

    # 5. Final Evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    final_score = validate(model, val_loader, device)
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_score}")

    # 6. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 7. Conditional Submission
    THRESHOLD = 0.47142532743789534
    if final_score < THRESHOLD:
        generate_submission(model, test_loader, device)
    else:
        print(
            f"\nValidation score ({final_score}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
