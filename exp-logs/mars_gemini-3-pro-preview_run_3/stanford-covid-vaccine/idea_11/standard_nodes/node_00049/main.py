import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, MCRMSELoss, mcrmse
from library.data import get_dataloaders
from library.model import RNAModel
from library.train import train_one_epoch, validate, generate_submission


def main():
    # ==============================
    # 1. Configuration & Setup
    # ==============================
    # Initialize config with overrides for a fast baseline
    config = Config(
        epochs=15,  # Limited epochs for speed
        batch_size=32,  # Moderate batch size
        debug=False,  # Use full dataset (it's small enough: ~2k samples)
        lr=1e-3,
        working_dir="./working",
        model_save_path="./working/best_model_runfile.pth",
    )

    # Ensure submission directory exists
    os.makedirs("./submission", exist_ok=True)
    os.makedirs(config.working_dir, exist_ok=True)

    seed_everything(config.seed)
    print(f"Device: {config.device}")

    # ==============================
    # 2. Data Loading
    # ==============================
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=True
    )

    # ==============================
    # 3. Model Initialization
    # ==============================
    model = RNAModel(config).to(config.device)

    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    scheduler = CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.eta_min
    )

    criterion = MCRMSELoss()

    # ==============================
    # 4. Training Loop
    # ==============================
    best_score = float("inf")

    print("Starting training...")
    for epoch in range(config.epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, config.device, config
        )

        # Validate
        val_score = validate(model, val_loader, config.device, config)

        # Scheduler Step
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{config.epochs} | Train Loss: {train_loss:.5f} | Val MCRMSE: {val_score:.5f}"
        )

        # Save Best Model
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), config.model_save_path)

    # ==============================
    # 5. Final Validation & Metric
    # ==============================
    print("Loading best model for final evaluation...")
    model.load_state_dict(
        torch.load(config.model_save_path, map_location=config.device)
    )
    model.eval()

    # We need to compute the metric on the full validation set again to be precise and for analysis
    # The 'validate' function returns the scalar score, but for failure analysis we need predictions.
    # We'll manually run inference on val_loader to get preds and targets.

    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(config.device)
            pair_index = batch["pair_index"].to(config.device)
            targets = batch["targets"].to(config.device)
            mask = batch["mask"].to(config.device)
            ids = batch["id"]

            preds = model(inputs, pair_index)  # (B, 107, 5)

            # We must apply the mask to get the scored positions for the metric
            # But for failure analysis per sample, we need to handle the flattening carefully.
            # Let's keep the structure (B, 68, 5) roughly.

            # The mask is 1 for first 68 positions, 0 otherwise.
            # We slice to config.pred_len (68)
            preds_sliced = preds[:, : config.pred_len, :]
            targets_sliced = targets[:, : config.pred_len, :]

            all_preds.append(preds_sliced.cpu().numpy())
            all_targets.append(targets_sliced.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate: (N_val, 68, 5)
    global_preds = np.concatenate(all_preds, axis=0)
    global_targets = np.concatenate(all_targets, axis=0)

    # Filter for scored columns
    global_preds = global_preds[..., config.SCORED_COLS_INDICES]
    global_targets = global_targets[..., config.SCORED_COLS_INDICES]

    # Compute Final Metric
    final_metric = mcrmse(global_targets, global_preds)
    print(f"Final Validation Metric: {final_metric}")

    # ==============================
    # 6. Failure Analysis
    # ==============================
    print("\nPerforming Failure Analysis...")

    # Calculate RMSE per sample
    # Shape: (N, 68, 5) -> (N,)
    # Mean squared error over (68, 5) then sqrt
    sample_mse = np.mean((global_targets - global_preds) ** 2, axis=(1, 2))
    sample_rmse = np.sqrt(sample_mse)

    # Create DataFrame for analysis
    analysis_df = pd.DataFrame({"id": all_ids, "sample_rmse": sample_rmse})

    # Load metadata to merge features
    val_meta_path = os.path.join(config.metadata_dir, "val.parquet")
    if os.path.exists(val_meta_path):
        val_meta_df = pd.read_parquet(val_meta_path)

        # Merge
        analysis_df = analysis_df.merge(val_meta_df, on="id", how="left")

        # Calculate correlations
        # We look for numerical columns of interest
        cols_to_correlate = ["signal_to_noise", "SN_filter", "seq_length"]

        # Also calculate mean errors if available
        if "reactivity_error" in val_meta_df.columns:
            # These are lists, calculate mean per sample
            analysis_df["mean_reactivity_error"] = analysis_df[
                "reactivity_error"
            ].apply(lambda x: np.mean(x) if isinstance(x, (list, np.ndarray)) else 0)
            cols_to_correlate.append("mean_reactivity_error")

        print("Correlation between Model Error (RMSE) and Features:")
        for col in cols_to_correlate:
            if col in analysis_df.columns:
                corr = analysis_df["sample_rmse"].corr(analysis_df[col])
                print(f"  {col}: {corr:.4f}")
    else:
        print("Validation metadata not found, skipping feature correlation.")

    # ==============================
    # 7. Submission
    # ==============================
    THRESHOLD = 0.7247761841173526

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        submission_df = generate_submission(model, test_loader, config.device, config)

        save_path = "./submission/submission.csv"
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
