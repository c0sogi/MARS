import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import provided library components
from library.config import Config
from library.dataset import RNADataset
from library.model import StabilizedWideBiGRU
from library.loss_metric import mcrmse
from library.runner import set_seed, train_epoch, validate, predict_and_submit


def evaluate_full(model, dataloader, device):
    """
    Computes MCRMSE on the full dataset to ensure exact precision.
    Averaging batch MCRMSEs is an approximation; this function concatenates
    all predictions and targets first.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in dataloader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            rwpe = batch["rwpe"].to(device)
            pair_enc = batch["pair_enc"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            preds = model(seq, loop, rwpe, pair_enc)

            all_preds.append(preds)
            all_targets.append(targets)
            all_masks.append(mask)

    # Concatenate all batches
    full_preds = torch.cat(all_preds, dim=0)
    full_targets = torch.cat(all_targets, dim=0)
    full_masks = torch.cat(all_masks, dim=0)

    # Compute metric using the library function
    score = mcrmse(full_preds, full_targets, full_masks)
    return score.item()


def run_failure_analysis(model, val_loader, val_df, device):
    """
    Analyzes model performance on the validation set.
    Computes RMSE per sample and correlates it with metadata features.
    """
    print("\nRunning Failure Analysis on Validation Set...")
    model.eval()

    all_sample_errors = []

    with torch.no_grad():
        for batch in val_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            rwpe = batch["rwpe"].to(device)
            pair_enc = batch["pair_enc"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            preds = model(seq, loop, rwpe, pair_enc)

            # Compute RMSE per sample
            # Squared error: (Batch, Seq, 3)
            squared_diff = (preds - targets) ** 2

            # Mask invalid positions
            mask_expanded = mask.unsqueeze(-1)
            masked_sq_diff = squared_diff * mask_expanded

            # Sum errors per sample: (Batch,)
            sum_sq_diff = masked_sq_diff.sum(dim=(1, 2))

            # Count valid elements per sample
            valid_counts = mask.sum(dim=1) * 3
            valid_counts = torch.clamp(valid_counts, min=1.0)

            mse_per_sample = sum_sq_diff / valid_counts
            rmse_per_sample = torch.sqrt(mse_per_sample)

            all_sample_errors.extend(rmse_per_sample.cpu().numpy())

    # Create analysis dataframe
    # Note: val_loader must be non-shuffled to align with val_df
    analysis_df = val_df.copy()

    if len(analysis_df) != len(all_sample_errors):
        print(
            f"Warning: Size mismatch in failure analysis ({len(analysis_df)} vs {len(all_sample_errors)}). Skipping."
        )
        return

    analysis_df["model_rmse"] = all_sample_errors

    # Compute derived features for correlation
    analysis_df["len_A"] = analysis_df["sequence"].apply(lambda x: x.count("A"))
    analysis_df["len_G"] = analysis_df["sequence"].apply(lambda x: x.count("G"))
    analysis_df["len_C"] = analysis_df["sequence"].apply(lambda x: x.count("C"))
    analysis_df["len_U"] = analysis_df["sequence"].apply(lambda x: x.count("U"))

    features_to_check = [
        "signal_to_noise",
        "SN_filter",
        "len_A",
        "len_G",
        "len_C",
        "len_U",
    ]

    print(f"{'Feature':<20} | {'Correlation with Error':<25}")
    print("-" * 50)

    for feat in features_to_check:
        if feat in analysis_df.columns:
            # Drop NaNs
            subset = analysis_df[[feat, "model_rmse"]].dropna()
            if len(subset) > 1:
                # Compute Pearson correlation
                corr = np.corrcoef(subset[feat], subset["model_rmse"])[0, 1]
                print(f"{feat:<20} | {corr:.4f}")


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Override Config for Fast Baseline execution
    Config.NUM_EPOCHS = 15
    Config.BATCH_SIZE = 32

    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("Loading datasets...")
    train_dataset = RNADataset(split="train", load_cached=True)
    val_dataset = RNADataset(split="val", load_cached=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Shuffle=False is critical for failure analysis alignment
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    print("Initializing model...")
    model = StabilizedWideBiGRU().to(device)

    optimizer = AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

    # --------------------------------------------------------------------------
    # 4. Training Loop
    # --------------------------------------------------------------------------
    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate (Approximate metric for monitoring)
        val_metric_approx = validate(model, val_loader, device)

        scheduler.step()

        # Save best model based on monitoring metric
        if val_metric_approx < best_mcrmse:
            best_mcrmse = val_metric_approx
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.5f} | Val MCRMSE (Approx): {val_metric_approx:.5f}"
        )

    # --------------------------------------------------------------------------
    # 5. Final Exact Evaluation
    # --------------------------------------------------------------------------
    print("\nComputing final exact validation metric...")
    # Load best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model not found. Using current model.")

    final_metric = evaluate_full(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # --------------------------------------------------------------------------
    # 6. Failure Analysis
    # --------------------------------------------------------------------------
    run_failure_analysis(model, val_loader, val_dataset.df, device)

    # --------------------------------------------------------------------------
    # 7. Conditional Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.6176461577

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} meets threshold {THRESHOLD}. Generating submission..."
        )
        # predict_and_submit handles test data loading and formatting
        # It will load the model from best_model_path
        predict_and_submit()
    else:
        print(
            f"\nMetric {final_metric} does not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
