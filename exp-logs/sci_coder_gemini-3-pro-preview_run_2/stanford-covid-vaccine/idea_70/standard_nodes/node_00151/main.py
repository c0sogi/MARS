import os
import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
from library.config import Config
from library.utils import set_seed, MCRMSELoss
from library.data import get_dataloaders
from library.model import RHIDFN
from library.train import train_one_epoch, validate, generate_submission


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates per-sample MCRMSE and correlates it with metadata features.
    """
    print("\nRunning Failure Analysis...")
    model.eval()

    all_errors = []

    # We need to ensure we align with the metadata.
    # val_loader is created with shuffle=False, so order is preserved.

    seq_scored = Config.SEQ_SCORED
    scored_indices = Config.SCORED_INDICES

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass (use refined prediction y2)
            _, y2 = model(inputs, partner_indices)

            # Calculate per-sample MCRMSE
            # 1. Mask Sequence
            preds_sliced = y2[:, :seq_scored, :]
            targets_sliced = targets[:, :seq_scored, :]

            # 2. Mask Columns
            preds_scored = preds_sliced[:, :, scored_indices]
            targets_scored = targets_sliced[:, :, scored_indices]

            # 3. MSE per sample (average over sequence and channels)
            # Shape: (Batch,)
            mse_per_sample = torch.mean(
                (preds_scored - targets_scored) ** 2, dim=(1, 2)
            )

            # 4. RMSE per sample
            rmse_per_sample = torch.sqrt(mse_per_sample)

            all_errors.extend(rmse_per_sample.cpu().numpy())

    # Load Metadata
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Add errors to dataframe
    # Ensure lengths match
    if len(all_errors) != len(val_df):
        print(
            f"Warning: Number of errors ({len(all_errors)}) does not match metadata length ({len(val_df)}). Skipping analysis."
        )
        return

    val_df["model_error"] = all_errors

    # Calculate Correlations
    features_to_check = ["signal_to_noise", "mean_reactivity"]

    print("-" * 40)
    print(f"{'Feature':<20} | {'Correlation':<10}")
    print("-" * 40)

    for feat in features_to_check:
        if feat in val_df.columns:
            # Handle potential NaNs
            valid_mask = val_df[feat].notna() & val_df["model_error"].notna()
            if valid_mask.sum() > 1:
                corr, _ = stats.pearsonr(
                    val_df.loc[valid_mask, feat], val_df.loc[valid_mask, "model_error"]
                )
                print(f"{feat:<20} | {corr:.4f}")
            else:
                print(f"{feat:<20} | N/A (Not enough data)")
        else:
            print(f"{feat:<20} | Not Found")
    print("-" * 40)


def main():
    # 1. Configuration Override for Fast Baseline
    Config.NUM_EPOCHS = 10  # Limit epochs for speed

    # 2. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 3. Data Loading
    print("Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(debug=False)

    # 4. Model & Optimization
    print("Initializing Model...")
    model = RHIDFN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = MCRMSELoss()

    # 5. Training Loop
    best_score = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler
        scheduler.step(val_score)

        # Save Best
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)

    # 6. Final Evaluation
    print("Training Complete.")

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Re-calculate metric on full validation set to be sure (and print required format)
    final_metric = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 8. Conditional Submission
    THRESHOLD = 0.47142532743789534

    if final_metric < THRESHOLD:
        print(
            f"Metric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        generate_submission(model, test_loader, device, submission_path)
        print(f"Submission saved to {submission_path}")
    else:
        print(
            f"Metric ({final_metric}) did not beat threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
