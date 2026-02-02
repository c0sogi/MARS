import os
import sys
import numpy as np
import pandas as pd
import torch
import scipy.stats as stats

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.loss import MCRMSELoss
from library.data import get_loaders
from library.model import TISRNModel
from library.train import train_one_epoch, validate


def analyze_failures(model, val_loader, config, val_csv_path):
    """
    Performs failure analysis by correlating per-sample errors with metadata features.
    """
    print("\nStarting Failure Analysis...")
    model.eval()

    all_errors = []
    # Scored indices corresponding to reactivity, deg_Mg_pH10, deg_Mg_50C
    scored_indices = [0, 1, 3]

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(config.device)
            partner_indices = batch["partner_indices"].to(config.device)
            targets = batch["targets"].to(config.device)
            mask = batch["mask"].to(config.device)

            # Forward Pass 2 (Refined predictions)
            _, preds = model(inputs, partner_indices, mask)

            # Slice scored columns
            preds_scored = preds[:, :, scored_indices]
            targets_scored = targets[:, :, scored_indices]

            # Squared Error: (B, L, 3)
            sq_diff = (preds_scored - targets_scored) ** 2

            # Apply mask
            mask_expanded = mask.unsqueeze(-1).float()
            sq_diff = sq_diff * mask_expanded

            # Mean over Length (valid positions only)
            valid_counts = mask.sum(dim=1).clamp(min=1.0)
            sum_sq_diff = sq_diff.sum(dim=1)  # (B, 3)
            mse_per_sample = sum_sq_diff / valid_counts.unsqueeze(-1)
            rmse_per_sample = torch.sqrt(mse_per_sample)

            # MCRMSE per sample: Mean over columns -> (B,)
            mcrmse_per_sample = rmse_per_sample.mean(dim=1)

            all_errors.extend(mcrmse_per_sample.cpu().numpy())

    all_errors = np.array(all_errors)

    # Load Metadata
    val_df = pd.read_csv(val_csv_path)

    # Ensure lengths match (robustness check)
    if len(all_errors) != len(val_df):
        n = min(len(all_errors), len(val_df))
        all_errors = all_errors[:n]
        val_df = val_df.iloc[:n]

    # Correlate with features
    features = ["signal_to_noise", "mean_reactivity", "SN_filter"]

    print(f"{'Feature':<20} | {'Correlation':<15}")
    print("-" * 40)

    for feat in features:
        if feat in val_df.columns:
            # Handle potential NaNs
            valid_mask = val_df[feat].notnull() & ~np.isnan(all_errors)
            if valid_mask.sum() > 1:
                corr, _ = stats.pearsonr(
                    val_df.loc[valid_mask, feat], all_errors[valid_mask]
                )
                print(f"{feat:<20} | {corr:.4f}")
            else:
                print(f"{feat:<20} | N/A")
        else:
            print(f"{feat:<20} | Not Found")


def generate_submission(model, test_loader, config):
    """
    Generates the submission file for the test set.
    """
    print("\nGenerating Submission...")
    model.eval()

    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(config.device)
            partner_indices = batch["partner_indices"].to(config.device)
            mask = batch["mask"].to(config.device)

            # Forward Pass 2
            _, preds = model(inputs, partner_indices, mask)

            # Collect predictions: (B, 107, 5)
            all_preds.append(preds.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)  # (N_samples, 107, 5)

    # Load Test Metadata for IDs
    test_df = pd.read_csv(config.test_csv)
    ids = test_df["id"].values

    submission_rows = []
    # Target columns order matches config and submission requirement
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Flatten predictions
    for i, sample_id in enumerate(ids):
        sample_preds = all_preds[i]  # (107, 5)
        # We must predict for all 107 positions
        for pos in range(config.seq_len):
            row_id = f"{sample_id}_{pos}"
            row_values = sample_preds[pos].tolist()
            submission_rows.append([row_id] + row_values)

    columns = ["id_seqpos"] + target_cols
    submission_df = pd.DataFrame(submission_rows, columns=columns)

    out_path = "./submission/submission.csv"
    submission_df.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path} with shape {submission_df.shape}")


def main():
    # 1. Configuration
    config = Config()
    # Override for fast baseline as per requirements
    config.epochs = 15

    seed_everything(config.seed)
    os.makedirs("./submission", exist_ok=True)

    # 2. Data Loading
    # Using cached data if available for speed
    train_loader, val_loader, test_loader = get_loaders(config, load_cached_data=True)

    # 3. Model Setup
    model = TISRNModel(config).to(config.device)
    criterion = MCRMSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )

    # 4. Training Loop
    best_mcrmse = float("inf")
    best_model_path = os.path.join(config.cache_dir, "best_model.pth")

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, config.device, config
        )
        val_loss, val_mcrmse = validate(model, val_loader, criterion, config.device)

        scheduler.step(val_mcrmse)

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)

    # 5. Final Validation
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=config.device))

    # Compute final metric
    _, final_mcrmse = validate(model, val_loader, criterion, config.device)

    # Required Output
    print(f"Final Validation Metric: {final_mcrmse}")

    # 6. Failure Analysis
    analyze_failures(model, val_loader, config, config.val_csv)

    # 7. Submission Logic
    THRESHOLD = 0.47142532743789534
    if final_mcrmse < THRESHOLD:
        generate_submission(model, test_loader, config)
    else:
        print(
            f"Validation metric {final_mcrmse} is not lower than {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
