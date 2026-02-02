import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import os
import sys

# Ensure library is in path
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, competition_metric
from library.data import get_loaders
from library.model import RNAModel
from library.train import train_epoch, validate, generate_submission


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override Config for Fast Baseline Execution
    Config.EPOCHS = 15
    Config.BATCH_SIZE = 32

    # Setup environment
    set_seed(Config.SEED)
    Config.setup()
    device = torch.device(Config.DEVICE)

    print(f"Execution Device: {device}")

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    print("Initializing Data Loaders...")
    # Using cached data if available, otherwise processes from scratch
    train_loader, val_loader, test_loader = get_loaders(debug=False)

    # =========================================================================
    # 3. Model Initialization
    # =========================================================================
    print("Initializing Model...")
    model = RNAModel(config=Config).to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # =========================================================================
    # 4. Training Loop
    # =========================================================================
    print("Starting Training...")
    best_score = float("inf")

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss, score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        print(
            f"Epoch {epoch}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | MCRMSE: {score:.6f}"
        )

        # Checkpoint
        if score < best_score:
            best_score = score
            torch.save(model.state_dict(), Config.MODEL_PATH)

    # =========================================================================
    # 5. Final Evaluation
    # =========================================================================
    print("Loading Best Model for Evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    # Recalculate metric on full validation set to be precise
    _, final_metric = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # =========================================================================
    # 6. Failure Analysis
    # =========================================================================
    print("\n==== Failure Analysis ====")

    # Load metadata to get features
    try:
        val_df = pd.read_parquet(Config.VAL_METADATA)

        # Generate predictions for validation set to compute per-sample error
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["inputs"].to(device)
                bpp_indices = batch["bpp_indices"].to(device)
                targets = batch["targets"].to(device)

                preds = model(inputs, bpp_indices)

                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

        preds_tensor = torch.cat(all_preds, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)

        # Slice to scored length (68) and scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C)
        # Config.SCORED_INDICES = [0, 1, 3]
        preds_scored = preds_tensor[:, : Config.PRED_LEN, Config.SCORED_INDICES]
        targets_scored = targets_tensor[:, : Config.PRED_LEN, Config.SCORED_INDICES]

        # Calculate MCRMSE per sample
        # MSE: (N, L, 3) -> Mean over L -> (N, 3)
        mse_per_sample = torch.mean((preds_scored - targets_scored) ** 2, dim=1)
        # RMSE: (N, 3)
        rmse_per_sample = torch.sqrt(mse_per_sample)
        # Mean RMSE: (N,)
        error_per_sample = torch.mean(rmse_per_sample, dim=1).numpy()

        # Attach to DataFrame (Assuming order is preserved, which it is for shuffle=False)
        if len(error_per_sample) == len(val_df):
            val_df["error"] = error_per_sample

            # Feature Engineering for Correlation
            val_df["pct_A"] = val_df["sequence"].apply(lambda x: x.count("A") / len(x))
            val_df["pct_G"] = val_df["sequence"].apply(lambda x: x.count("G") / len(x))
            val_df["pct_C"] = val_df["sequence"].apply(lambda x: x.count("C") / len(x))
            val_df["pct_U"] = val_df["sequence"].apply(lambda x: x.count("U") / len(x))
            val_df["pct_unpaired"] = val_df["structure"].apply(
                lambda x: x.count(".") / len(x)
            )

            # Select features
            features = [
                "signal_to_noise",
                "SN_filter",
                "pct_A",
                "pct_G",
                "pct_C",
                "pct_U",
                "pct_unpaired",
            ]

            # Calculate Correlation
            correlations = (
                val_df[features].corrwith(val_df["error"]).sort_values(ascending=False)
            )

            print("Correlation between Error and Input Features:")
            print(correlations)
        else:
            print(
                f"Warning: Validation set size mismatch. Preds: {len(error_per_sample)}, DF: {len(val_df)}"
            )

    except Exception as e:
        print(f"Failure analysis failed: {e}")

    # =========================================================================
    # 7. Submission Generation
    # =========================================================================
    THRESHOLD = 0.5978901386

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader, device)
    else:
        print(
            f"\nMetric ({final_metric}) is NOT below threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
