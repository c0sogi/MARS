import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import set_seed, get_score
from library.loss import MCRMSELoss
from library.data import get_loaders
from library.model import HCTADPBiGRU
from library.train import train_one_epoch, validate, generate_submission


def analyze_failures(model, val_loader, device, val_metadata_path):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between model error and input features.
    """
    print("\nRunning Failure Analysis...")
    model.eval()

    all_losses = []
    all_ids = []

    criterion = torch.nn.MSELoss(reduction="none")  # To get per-element error

    # 1. Collect errors per sample
    with torch.no_grad():
        for features, pair_indices, pair_masks, targets, sample_ids in val_loader:
            features = features.to(device)
            pair_indices = pair_indices.to(device)
            pair_masks = pair_masks.to(device)
            targets = targets.to(device)

            outputs = model(features, pair_indices, pair_masks)

            # Slice to scored length (68)
            scored_len = targets.shape[1]
            outputs_sliced = outputs[:, :scored_len, :]

            # Calculate MSE per sample (average over sequence and targets)
            # Shape: (B, 68, 5) -> mean over (1, 2) -> (B,)
            loss = criterion(outputs_sliced, targets).mean(dim=(1, 2))

            all_losses.extend(loss.cpu().numpy())
            all_ids.extend(sample_ids)

    # Create Error DataFrame
    error_df = pd.DataFrame(
        {"id": all_ids, "mse": all_losses, "rmse": np.sqrt(all_losses)}
    )

    # 2. Load Metadata for features
    if not os.path.exists(val_metadata_path):
        print(
            f"Warning: Validation metadata not found at {val_metadata_path}. Skipping detailed analysis."
        )
        return

    meta_df = pd.read_parquet(val_metadata_path)

    # Merge errors with metadata
    analysis_df = pd.merge(error_df, meta_df, on="id", how="left")

    # 3. Feature Engineering for Correlation
    # GC Content
    analysis_df["gc_content"] = analysis_df["sequence"].apply(
        lambda s: (s.count("G") + s.count("C")) / len(s)
    )
    # Paired Percentage
    analysis_df["paired_pct"] = analysis_df["structure"].apply(
        lambda s: (s.count("(") + s.count(")")) / len(s)
    )

    # Features to check
    features_to_check = ["signal_to_noise", "SN_filter", "gc_content", "paired_pct"]

    print("-" * 40)
    print(f"{'Feature':<20} | {'Correlation (r)':<15} | {'P-value':<10}")
    print("-" * 40)

    for feat in features_to_check:
        if feat in analysis_df.columns:
            # Drop NaNs just in case
            valid_data = analysis_df[[feat, "rmse"]].dropna()
            if len(valid_data) > 1:
                r, p = pearsonr(valid_data[feat], valid_data["rmse"])
                print(f"{feat:<20} | {r:<15.4f} | {p:<10.4g}")
            else:
                print(f"{feat:<20} | Not enough data")
        else:
            print(f"{feat:<20} | Not found in metadata")
    print("-" * 40)


def main():
    # 1. Setup and Configuration
    # Modify Config for Fast Baseline
    Config.epochs = 10  # Limit epochs for speed
    Config.batch_size = 32
    Config.patience = 3  # Strict early stopping

    # Ensure submission directory exists
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_filepath = os.path.join(submission_dir, "submission.csv")

    config = Config()
    set_seed(config.seed)
    device = torch.device(config.device)

    print(f"Device: {device}")
    print(f"Fast Baseline Config: Epochs={config.epochs}, Batch={config.batch_size}")

    # 2. Data Loading
    print("Loading Data...")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing Model...")
    model = HCTADPBiGRU().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.eta_min
    )
    criterion = MCRMSELoss()

    # 5. Training Loop
    best_mcrmse = float("inf")

    print("Starting Training...")
    for epoch in range(config.epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config.max_grad_norm
        )

        # Validate
        val_loss, val_mcrmse = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{config.epochs} | Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.4f} | Val MCRMSE: {val_mcrmse:.6f}"
        )

        # Checkpointing
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), config.best_model_path)

    # 6. Final Evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(config.best_model_path, map_location=device))

    # Compute final metric on validation set
    _, final_val_mcrmse = validate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_val_mcrmse}")

    # 7. Failure Analysis
    analyze_failures(model, val_loader, device, config.val_file)

    # 8. Submission Logic
    THRESHOLD = 0.5884495377540588
    if final_val_mcrmse < THRESHOLD:
        print(
            f"\nValidation metric ({final_val_mcrmse}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader, device, submission_filepath)
    else:
        print(
            f"\nValidation metric ({final_val_mcrmse}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
