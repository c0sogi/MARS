import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scipy.stats as stats
from torch.utils.data import DataLoader

# Import components from the provided library files
from library.config import Config, seed_everything
from library.data import load_data, RNADataset
from library.model import RNAModel
from library.train import train_one_epoch, validate
from library.utils import format_submission


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running on device: {device}")
    print(f"Training for {Config.EPOCHS} epochs...")

    # 2. Data Loading
    # load_data handles caching and processing
    train_ids, train_seq, train_loop, train_dist, train_tgt = load_data("train")
    val_ids, val_seq, val_loop, val_dist, val_tgt = load_data("val")

    train_dataset = RNADataset(train_seq, train_loop, train_dist, train_tgt)
    val_dataset = RNADataset(val_seq, val_loop, val_dist, val_tgt)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = RNAModel(Config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    criterion = nn.MSELoss()

    # 4. Training Loop
    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        val_mcrmse = validate(model, val_loader, device)

        # Simple logging
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_mcrmse:.6f}"
        )

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)

    # 5. Final Evaluation & Failure Analysis
    print("\nPerforming Final Evaluation and Failure Analysis...")

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Run inference on validation set to get raw predictions for analysis
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            seq = batch["sequence"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["distance"].to(device)
            target = batch["target"].to(device)

            pred = model(seq, loop, dist)
            pred_scored = pred[:, : Config.SEQ_SCORED, :]

            val_preds.append(pred_scored.cpu().numpy())
            val_targets.append(target.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Calculate Final Metric
    mse_per_col = np.mean((val_targets - val_preds) ** 2, axis=(0, 1))
    rmse_per_col = np.sqrt(mse_per_col)
    final_metric = np.mean(rmse_per_col)

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate error per sample: Mean of RMSEs across the 3 columns for that sample
    # Shape: (N, 68, 3) -> MSE per sample per col (mean over axis 1) -> Sqrt -> Mean over axis 1 (cols)
    sample_mse = np.mean((val_targets - val_preds) ** 2, axis=1)  # (N, 3)
    sample_rmse = np.sqrt(sample_mse)  # (N, 3)
    sample_error = np.mean(sample_rmse, axis=1)  # (N,)

    # Load metadata to get features
    if os.path.exists(Config.VAL_METADATA):
        val_meta_df = pd.read_parquet(Config.VAL_METADATA)

        # Ensure alignment (assuming load_data preserves order, which it does based on implementation)
        # Check for features
        analysis_features = ["signal_to_noise", "SN_filter", "reads", "seq_length"]
        print("\nFailure Analysis (Correlation with Error):")
        for feat in analysis_features:
            if feat in val_meta_df.columns:
                # Handle potential NaNs or non-numeric types if any (though data seems clean)
                feat_values = val_meta_df[feat].values
                if len(feat_values) == len(sample_error):
                    try:
                        corr, _ = stats.pearsonr(feat_values, sample_error)
                        print(f"  {feat}: {corr:.4f}")
                    except Exception as e:
                        print(f"  {feat}: Could not calculate correlation ({e})")
    else:
        print("Validation metadata not found, skipping feature correlation analysis.")

    # 6. Submission Generation
    THRESHOLD = 0.6176461577

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric {final_metric:.6f} passed threshold {THRESHOLD}. Generating submission..."
        )

        # Load Test Data
        test_ids, test_seq, test_loop, test_dist = load_data("test")
        test_dataset = RNADataset(test_seq, test_loop, test_dist, None)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        all_test_preds = []
        with torch.no_grad():
            for batch in test_loader:
                seq = batch["sequence"].to(device)
                loop = batch["loop"].to(device)
                dist = batch["distance"].to(device)

                pred = model(seq, loop, dist)  # (B, 107, 3)
                all_test_preds.append(pred.cpu().numpy())

        all_test_preds = np.concatenate(all_test_preds, axis=0)

        # Format and Save
        format_submission(test_ids, all_test_preds, Config.SUBMISSION_PATH)
    else:
        print(
            f"\nValidation metric {final_metric:.6f} did not pass threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
