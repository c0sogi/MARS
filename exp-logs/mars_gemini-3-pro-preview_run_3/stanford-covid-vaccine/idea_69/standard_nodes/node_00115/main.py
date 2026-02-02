import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, get_device, calculate_mcrmse
from library.loss import MCRMSELoss
from library.data import get_loaders
from library.model import RNAModel
from library.train import train_one_epoch, evaluate, generate_submission

# =============================================================================
# Configuration Overrides for Fast Baseline
# =============================================================================
# Limit epochs and patience to ensure execution within time limits
Config.NUM_EPOCHS = 10
Config.PATIENCE = 3
# Ensure we use the full dataset for meaningful results, but rely on early stopping
Config.DEBUG = False


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates per-sample error and correlates it with metadata features.
    """
    print("\n==== Failure Analysis ====")
    model.eval()

    all_preds = []
    all_targets = []
    all_ids = []

    # 1. Collect Predictions and Targets
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            targets = batch["targets"]  # Keep on CPU
            ids = batch["ids"]

            preds = model(inputs, pair_indices, pair_masks)

            all_preds.append(preds.detach().cpu())
            all_targets.append(targets)
            all_ids.extend(ids)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # 2. Calculate Per-Sample Error (MCRMSE for that sample)
    # Slice to scored length
    seq_scored = Config.PRED_LEN
    preds_sliced = all_preds[:, :seq_scored, :]

    # Identify scored columns indices
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]

    preds_filtered = preds_sliced[:, :, scored_indices]
    targets_filtered = all_targets[:, :, scored_indices]

    # MSE per sample: Average over sequence (dim 1) and columns (dim 2)
    # Result shape: (N,)
    mse_per_sample = torch.mean((targets_filtered - preds_filtered) ** 2, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # 3. Load Metadata to get features
    val_meta_path = Config.VAL_PARQUET
    if not os.path.exists(val_meta_path):
        print("Validation metadata not found. Skipping detailed metadata correlation.")
        return

    val_df = pd.read_parquet(val_meta_path)

    # Create a DataFrame for analysis
    analysis_df = pd.DataFrame({"id": all_ids, "error": rmse_per_sample})

    # Merge with metadata
    merged_df = pd.merge(analysis_df, val_df, on="id", how="left")

    # 4. Compute Correlations
    features_to_check = ["signal_to_noise", "SN_filter", "seq_length"]
    # Also check nucleotide content if possible, but let's stick to available columns

    print("Correlation between Model Error (RMSE) and Input Features:")
    for feat in features_to_check:
        if feat in merged_df.columns:
            # Drop NaNs just in case
            valid_data = merged_df[[feat, "error"]].dropna()
            if len(valid_data) > 0:
                corr = valid_data[feat].corr(valid_data["error"])
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: No valid data")
        else:
            print(f"  {feat}: Not found in metadata")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    # load_cached_data=True to use any pre-existing processed files
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing High-Capacity GLU-Refined BiGRU Model...")
    model = RNAModel(config=Config).to(device)

    # 4. Training Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Adjust T_max to the overridden epoch count
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.ETA_MIN
    )

    criterion = MCRMSELoss()

    # 5. Training Loop
    best_score = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = evaluate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_score:.6f} | "
            f"LR: {current_lr:.2e}"
        )

        # Checkpointing
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            # print(f"  Saved best model.")
        else:
            patience_counter += 1
            # print(f"  Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Final Evaluation
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    final_val_score = evaluate(model, val_loader, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_val_score}")

    # 7. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 8. Conditional Submission
    THRESHOLD = 0.5884495377540588

    if final_val_score < THRESHOLD:
        print(
            f"Validation score ({final_val_score}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader, device)
    else:
        print(
            f"Validation score ({final_val_score}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
