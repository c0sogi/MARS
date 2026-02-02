import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, calculate_mcrmse
from library.data import get_dataloaders
from library.model import DeepBiGRUNet
from library.train import train_one_epoch, evaluate, generate_submission

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model performance on the validation set by correlating
    error magnitudes with sample metadata features.
    """
    model.eval()

    # 1. Collect Sample-wise Errors
    ids = []
    sample_rmses = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            targets = batch["targets"].to(device)
            batch_ids = batch["id"]

            # Forward pass
            preds = model(features, pair_indices, pair_masks)

            # Slice to scored length
            preds = preds[:, : Config.SEQ_SCORED, :]
            targets = targets[:, : Config.SEQ_SCORED, :]

            # Calculate RMSE per sample (average over sequence and channels)
            # Shape: (B, SeqScored, 5) -> (B,)
            mse = torch.mean((preds - targets) ** 2, dim=(1, 2))
            rmse = torch.sqrt(mse)

            ids.extend(batch_ids)
            sample_rmses.extend(rmse.cpu().numpy())

    # 2. Load Metadata
    val_meta_path = Config.VAL_METADATA_PATH
    if not os.path.exists(val_meta_path):
        print("Validation metadata not found. Skipping detailed failure analysis.")
        return

    df_val = pd.read_parquet(val_meta_path)

    # 3. Create Analysis DataFrame
    error_df = pd.DataFrame({"id": ids, "rmse": sample_rmses})

    # Merge with metadata
    analysis_df = pd.merge(error_df, df_val, on="id", how="inner")

    # 4. Feature Engineering for Correlation
    # Calculate nucleotide content
    for char in ["A", "G", "C", "U"]:
        analysis_df[f"pct_{char}"] = analysis_df["sequence"].apply(
            lambda s: s.count(char) / len(s)
        )

    # 5. Compute Correlations
    correlations = {}
    features_to_check = [
        "signal_to_noise",
        "SN_filter",
        "pct_A",
        "pct_G",
        "pct_C",
        "pct_U",
    ]

    print("\nFailure Analysis (Correlation with Error):")
    for feat in features_to_check:
        if feat in analysis_df.columns:
            # Ensure numeric
            if pd.api.types.is_numeric_dtype(analysis_df[feat]):
                corr, _ = pearsonr(analysis_df["rmse"], analysis_df[feat])
                correlations[feat] = corr
                print(f"  {feat}: {corr:.4f}")
            else:
                # Handle categorical like SN_filter if it's not numeric
                try:
                    corr, _ = pearsonr(
                        analysis_df["rmse"], analysis_df[feat].astype(float)
                    )
                    correlations[feat] = corr
                    print(f"  {feat}: {corr:.4f}")
                except:
                    pass


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Adjust Config for Fast Baseline
    # 25 Epochs is sufficient for convergence on this small dataset (~1700 samples)
    # while keeping runtime low.
    Config.EPOCHS = 25

    # 2. Data Loading
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = DeepBiGRUNet().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.EPOCHS,  # Adjusted to match new epoch count
        eta_min=Config.ETA_MIN,
    )

    # 5. Training Loop
    best_mcrmse = float("inf")
    best_model_path = Config.MODEL_SAVE_PATH

    # Ensure working directory exists
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_mcrmse = evaluate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        # Checkpoint
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)

    # 6. Final Evaluation
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Compute Final Metric
    final_metric = evaluate(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 8. Submission
    threshold = 0.5978901386
    if final_metric < threshold:
        # Create submission directory
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        # Generate submission
        generate_submission(model, test_loader, device, submission_path)


if __name__ == "__main__":
    main()
