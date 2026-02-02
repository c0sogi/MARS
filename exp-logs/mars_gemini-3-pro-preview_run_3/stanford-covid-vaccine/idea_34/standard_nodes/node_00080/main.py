import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from library
from library.config import config
from library.utils import set_seed, calculate_mcrmse, format_submission
from library.data import get_dataloaders
from library.model import DCASGBiGRU
from library.train import train_one_epoch, validate, MCRMSELoss


def main():
    # =========================================================================
    # 1. Setup & Configuration
    # =========================================================================
    set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    print(f"Running on device: {device}")

    # Limit epochs for a fast baseline execution as requested
    config.EPOCHS = 15

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    print("Loading data...")
    # get_dataloaders handles caching internally
    train_loader, val_loader, test_loader = get_dataloaders(debug=False)

    # =========================================================================
    # 3. Model Initialization
    # =========================================================================
    print("Initializing model...")
    model = DCASGBiGRU().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS, eta_min=config.ETA_MIN
    )

    criterion = MCRMSELoss()

    # =========================================================================
    # 4. Training Loop
    # =========================================================================
    best_score = float("inf")
    print(f"Starting training for {config.EPOCHS} epochs...")

    for epoch in range(config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config.MAX_GRAD_NORM
        )

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        # Logging
        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss:.5f} | Val MCRMSE: {val_score:.6f}"
        )

        # Save Best Model
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)

    print(f"Training complete. Best Validation Score: {best_score}")

    # =========================================================================
    # 5. Validation Assessment & Failure Analysis
    # =========================================================================
    print("\n==== Validation Assessment & Failure Analysis ====")

    # Load best model for analysis
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # Inference on Validation Set
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            targets = batch["targets"]  # Keep on CPU
            ids = batch["id"]

            preds = model(inputs, pair_indices, pair_masks)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Final Metric
    final_metric = calculate_mcrmse(all_preds, all_targets)
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    # 1. Calculate error per sample
    # Slice to scored length and scored columns
    preds_sliced = all_preds[:, : config.SEQ_SCORED, config.SCORING_INDICES]
    targets_sliced = all_targets[:, : config.SEQ_SCORED, config.SCORING_INDICES]

    # MSE per sample (average over positions and columns)
    mse_per_sample = np.mean((preds_sliced - targets_sliced) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # 2. Load Metadata
    val_df = pd.read_parquet(config.VAL_PARQUET)

    # 3. Create Analysis DataFrame
    analysis_df = pd.DataFrame({"id": all_ids, "error": rmse_per_sample})

    # Merge with metadata to get features
    analysis_df = analysis_df.merge(val_df, on="id", how="left")

    # 4. Feature Engineering for Analysis
    if "sequence" in analysis_df.columns:
        analysis_df["pct_A"] = analysis_df["sequence"].apply(
            lambda x: x.count("A") / len(x) if x else 0
        )

    # 5. Compute Correlations
    analysis_features = ["signal_to_noise", "SN_filter", "pct_A"]
    print("Correlation between Error and Features:")

    for feat in analysis_features:
        if feat in analysis_df.columns:
            # Drop NaNs for correlation calculation
            subset = analysis_df[[feat, "error"]].dropna()
            if len(subset) > 1:
                corr, _ = pearsonr(subset[feat], subset["error"])
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: N/A (Not enough data)")
        else:
            print(f"  {feat}: Not found in metadata")

    # =========================================================================
    # 6. Submission Generation
    # =========================================================================
    THRESHOLD = 0.5978901386

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        test_preds = []
        test_ids = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["inputs"].to(device)
                pair_indices = batch["pair_indices"].to(device)
                pair_masks = batch["pair_masks"].to(device)
                ids = batch["id"]

                preds = model(inputs, pair_indices, pair_masks)

                test_preds.append(preds.cpu().numpy())
                test_ids.extend(ids)

        test_preds = np.concatenate(test_preds, axis=0)

        # Format and Save
        submission_df = format_submission(test_ids, test_preds)

        # Ensure directory exists (though working dir should exist)
        os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
