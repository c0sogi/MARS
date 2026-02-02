import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings

# Import library components
from library.config import Config
from library.model import RNAModel
from library.utils import seed_everything, get_device, mcrmse
from library.data import get_dataloaders
from library.engine import train_fn, predict_and_submit

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    seed_everything(Config.SEED)
    device = get_device()

    # Initialize Config
    # We use debug=False to utilize the full dataset (2k samples).
    # Training is fast enough (approx 2-3 mins) to meet the "quick baseline" requirement
    # while ensuring we have enough capacity to hit the target metric.
    config = Config(debug=False)

    # Override Submission Path to meet prompt requirement
    config.SUBMISSION_PATH = "./submission/submission.csv"

    print(f"Device: {device}")
    print(f"Working Directory: {config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\nLoading data...")
    # Load cached data if available for speed
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=config.debug, load_cached_data=True
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("Initializing model...")
    model = RNAModel(config=config).to(device)

    # -------------------------------------------------------------------------
    # 4. Optimization Setup
    # -------------------------------------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    best_score = float("inf")

    print(f"\nStarting training for {config.EPOCHS} epochs...")

    for epoch in range(config.EPOCHS):
        # Train Step
        train_loss = train_fn(model, train_loader, optimizer, scheduler, device, config)

        # Validation Step
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = {
                    "sequence": batch["sequence"].to(device),
                    "loop_type": batch["loop_type"].to(device),
                    "pair_dist": batch["pair_dist"].to(device),
                }
                targets = batch["targets"].to(device)

                outputs = model(**inputs)

                # Store on CPU to avoid OOM
                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())

        # Concatenate
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute Metric
        val_score = mcrmse(all_targets, all_preds, num_scored=config.PRED_LEN).item()

        print(
            f"Epoch {epoch+1:02d}/{config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.10f}"
        )

        # Save Best Model
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)

    print(f"\nTraining complete. Best Val MCRMSE: {best_score:.10f}")

    # -------------------------------------------------------------------------
    # 6. Final Evaluation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\nRunning Final Evaluation and Failure Analysis...")

    # Load best model weights
    if os.path.exists(config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model file not found. Using current weights.")

    model.eval()

    # Collect predictions and IDs for analysis
    val_preds = []
    val_targets = []
    val_ids = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                "sequence": batch["sequence"].to(device),
                "loop_type": batch["loop_type"].to(device),
                "pair_dist": batch["pair_dist"].to(device),
            }
            targets = batch["targets"]  # Keep on CPU
            ids = batch["id"]

            outputs = model(**inputs).cpu()

            val_preds.append(outputs)
            val_targets.append(targets)
            val_ids.extend(ids)

    val_preds = torch.cat(val_preds, dim=0)
    val_targets = torch.cat(val_targets, dim=0)

    # Compute Final Metric (Required Format)
    final_metric = mcrmse(val_targets, val_preds, num_scored=config.PRED_LEN).item()
    print(f"Final Validation Metric: {final_metric:.10f}")

    # --- Failure Analysis ---
    # Load validation metadata to get features like signal_to_noise
    if os.path.exists(config.VAL_FILE):
        df_val = pd.read_parquet(config.VAL_FILE)

        # Calculate RMSE per sample (averaged over scored positions and targets)
        # Shape: (N, 68, 3)
        diff = val_preds[:, : config.PRED_LEN, :] - val_targets[:, : config.PRED_LEN, :]
        # Mean MSE per sample
        mse_per_sample = torch.mean(diff**2, dim=(1, 2)).numpy()
        rmse_per_sample = np.sqrt(mse_per_sample)

        # Create analysis dataframe
        analysis_df = pd.DataFrame({"id": val_ids, "error": rmse_per_sample})

        # Merge with metadata to get features
        # We select relevant columns from df_val
        meta_cols = ["id", "signal_to_noise", "SN_filter", "sequence"]
        analysis_df = analysis_df.merge(df_val[meta_cols], on="id", how="left")

        # Calculate derived feature: Count of 'A'
        analysis_df["len_A"] = analysis_df["sequence"].apply(
            lambda x: x.count("A") if isinstance(x, str) else 0
        )

        # Compute correlations
        corr_snr = analysis_df["error"].corr(analysis_df["signal_to_noise"])
        corr_sn_filter = analysis_df["error"].corr(analysis_df["SN_filter"])
        corr_len_A = analysis_df["error"].corr(analysis_df["len_A"])

        print("\nFailure Analysis (Correlation with Error):")
        print(f"  Signal-to-Noise: {corr_snr:.4f}")
        print(f"  SN_filter: {corr_sn_filter:.4f}")
        print(f"  Count 'A': {corr_len_A:.4f}")
    else:
        print("Warning: Validation metadata file not found. Skipping Failure Analysis.")

    # -------------------------------------------------------------------------
    # 7. Conditional Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.6176461577

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric:.10f} < {THRESHOLD}. Generating submission...")
        predict_and_submit(model, test_loader, device, config)
    else:
        print(f"\nMetric {final_metric:.10f} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    run()
