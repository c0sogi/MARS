import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.dataset import get_dataset
from library.model import RNAResidualBiGRU
from library.loss import MaskedMSELoss, mcrmse
from library.train import train_one_epoch, validate
from library.predict import predict_and_format


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_failure_analysis(model, val_loader, device):
    """
    Analyzes model performance on the validation set to identify error patterns.
    """
    print("\nRunning Failure Analysis...")
    model.eval()

    all_errors = []
    feature_stats = []

    # Nuc map: A:0, G:1, C:2, U:3

    with torch.no_grad():
        for batch in val_loader:
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            pair_dist = batch["pair_dist"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            # Forward pass
            preds = model(sequence, loop_type, pair_dist)

            # Calculate error per sample
            # Error metric: RMSE over valid positions and scored columns
            # (B, L, C)
            squared_diff = (preds - targets) ** 2

            # Apply mask (B, L) -> (B, L, 1)
            mask_expanded = mask.unsqueeze(-1)
            masked_squared_diff = squared_diff * mask_expanded.float()

            # Sum over Length and Channels for each sample
            # Denominator: Count of valid positions * channels per sample
            # Note: All samples have the same mask structure in this specific dataset (first 68 bases),
            # but we calculate generically.
            sum_sq_diff = masked_squared_diff.sum(dim=(1, 2))  # (B,)
            count_valid = mask_expanded.sum(dim=(1, 2))  # (B,)

            # Avoid div by zero
            count_valid = torch.clamp(count_valid, min=1.0)
            mse_per_sample = sum_sq_diff / count_valid
            rmse_per_sample = torch.sqrt(mse_per_sample)

            all_errors.extend(rmse_per_sample.cpu().tolist())

            # Extract Features for correlation
            # sequence: (B, L)
            seq_cpu = sequence.cpu()
            pair_cpu = pair_dist.cpu()

            for i in range(len(seq_cpu)):
                s = seq_cpu[i]
                p = pair_cpu[i]

                # Features
                len_A = (s == 0).sum().item()
                len_G = (s == 1).sum().item()
                len_C = (s == 2).sum().item()
                len_U = (s == 3).sum().item()
                # Paired bases have non-zero distance
                n_paired = (p != 0).sum().item()

                feature_stats.append(
                    {
                        "len_A": len_A,
                        "len_G": len_G,
                        "len_C": len_C,
                        "len_U": len_U,
                        "n_paired": n_paired,
                        "gc_content": (len_G + len_C) / 107.0,
                    }
                )

    # Create DataFrame
    df_analysis = pd.DataFrame(feature_stats)
    df_analysis["error"] = all_errors

    # Compute Correlations
    correlations = (
        df_analysis.corr()["error"].drop("error").sort_values(ascending=False)
    )

    print("Correlation between Input Features and Model Error (RMSE):")
    print(correlations)

    return correlations


def main():
    # 1. Configuration
    Config.setup()
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    train_dataset = get_dataset("train", load_cached_data=True)
    val_dataset = get_dataset("val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = RNAResidualBiGRU().to(device)

    criterion = MaskedMSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # 4. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")
    best_mcrmse = float("inf")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, Config.MAX_GRAD_NORM
        )

        # Validate
        val_mcrmse = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss:.5f} | Val MCRMSE: {val_mcrmse:.5f}"
        )

        # Save Best
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            # print(f"  Saved best model.")

    # 5. Final Validation on Best Model
    print("\nLoading best model for final evaluation...")
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    final_metric = validate(model, val_loader, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 7. Submission
    # Threshold check: 0.6226052641868591
    THRESHOLD = 0.6226052641868591

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} is better than threshold {THRESHOLD}. Generating submission..."
        )
        predict_and_format(device=Config.DEVICE)
    else:
        print(
            f"\nMetric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
