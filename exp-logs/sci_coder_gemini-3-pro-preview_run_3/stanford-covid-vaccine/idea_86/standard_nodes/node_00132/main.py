import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, calculate_mcrmse
from library.loss import MCRMSELoss
from library.data import get_dataloaders
from library.model import RNAModel
from library.train import train_one_epoch, validate, generate_submission


def analyze_failures(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlations between error magnitude and input features.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_ids = []

    # 1. Collect Predictions and Targets
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            outputs = model(inputs, pair_indices, pair_mask)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
            all_ids.extend(ids)

    full_preds = torch.cat(all_preds, dim=0)
    full_targets = torch.cat(all_targets, dim=0)

    # 2. Calculate Sample-wise RMSE (Error Magnitude)
    # Slice to scored region
    seq_scored = Config.SEQ_SCORED
    preds_sliced = full_preds[:, :seq_scored, :]
    targets_sliced = full_targets[:, :seq_scored, :]

    # Filter for scored columns
    target_cols = Config.TARGET_COLS
    scored_cols = Config.SCORED_COLS
    scored_indices = [i for i, col in enumerate(target_cols) if col in scored_cols]

    # Calculate MSE per sample (average over sequence and scored columns)
    # Shape: (N, Seq, Cols)
    diff = preds_sliced[:, :, scored_indices] - targets_sliced[:, :, scored_indices]
    mse_per_sample = torch.mean(diff**2, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # Create Error DataFrame
    error_df = pd.DataFrame({"id": all_ids, "rmse": rmse_per_sample})

    # 3. Load Metadata for Features
    val_meta_df = pd.read_parquet(Config.VAL_DATA_PATH)

    # Merge
    analysis_df = pd.merge(error_df, val_meta_df, on="id", how="left")

    # 4. Feature Engineering for Analysis
    # Nucleotide content
    for char in ["A", "G", "C", "U"]:
        analysis_df[f"pct_{char}"] = analysis_df["sequence"].apply(
            lambda s: s.count(char) / len(s)
        )

    # Structure content
    analysis_df["pct_unpaired"] = analysis_df["structure"].apply(
        lambda s: s.count(".") / len(s)
    )

    # 5. Calculate Correlations
    features = [
        "signal_to_noise",
        "SN_filter",
        "pct_A",
        "pct_G",
        "pct_C",
        "pct_U",
        "pct_unpaired",
    ]
    print("\n==== Failure Analysis (Correlation with Error) ====")
    print(f"{'Feature':<20} {'Correlation':<10}")
    print("-" * 30)

    for feat in features:
        if feat in analysis_df.columns:
            corr = analysis_df[feat].corr(analysis_df["rmse"])
            print(f"{feat:<20} {corr:.4f}")
    print("=================================================")


def main():
    # 1. Setup
    # Override Config for Fast Baseline
    Config.EPOCHS = 15  # Sufficient for small dataset
    Config.setup()
    set_seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, batch_size=Config.BATCH_SIZE
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = RNAModel().to(device)

    criterion = MCRMSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # 4. Training Loop
    best_mcrmse = float("inf")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_mcrmse = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        print(
            f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_mcrmse:.6f}"
        )

        # Save Best Model
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    # 5. Final Evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # Re-run validation to ensure we have the exact metric of the loaded model
    final_val_metric = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_val_metric}")

    # 6. Failure Analysis
    analyze_failures(model, val_loader, device)

    # 7. Submission
    THRESHOLD = 0.5884495377540588

    if final_val_metric < THRESHOLD:
        print(
            f"\nMetric ({final_val_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"\nMetric ({final_val_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
