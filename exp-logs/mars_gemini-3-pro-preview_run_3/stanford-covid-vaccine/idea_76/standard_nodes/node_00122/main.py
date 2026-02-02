import os
import time
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided library files
from library.config import Config
from library.utils import set_seed, mcrmse
from library.data import get_dataloaders
from library.model import HCTDBiGRU
from library.train import train_epoch, validate


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override Config for fast baseline execution
    Config.EPOCHS = 15
    Config.PATIENCE = 5
    Config.SUBMISSION_PATH = "./submission/submission.csv"

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    print("Initializing DataLoaders...")
    # Using load_cached_data=True to utilize preprocessed files in ./working
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
        debug=False,
    )

    # =========================================================================
    # 3. Model Initialization
    # =========================================================================
    print("Initializing Model (HC-TD-BiGRU)...")
    model = HCTDBiGRU().to(device)

    # =========================================================================
    # 4. Optimization
    # =========================================================================
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # Loss function wrapper using mcrmse (training uses all columns)
    def criterion(y_true, y_pred):
        return mcrmse(y_true, y_pred, scored_indices=None)

    # =========================================================================
    # 5. Training Loop
    # =========================================================================
    best_val_score = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion)

        # Validate (Calculates MCRMSE on scored columns only)
        val_score = validate(model, val_loader, device)

        # Scheduler step
        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score} | Time: {elapsed:.2f}s"
        )

        # Checkpointing & Early Stopping
        if val_score < best_val_score:
            best_val_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # =========================================================================
    # 6. Final Evaluation & Failure Analysis
    # =========================================================================
    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Re-calculate final metric to ensure accuracy and print required format
    final_val_score = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_val_score}")

    # --- Failure Analysis ---
    print("\nPerforming Failure Analysis...")

    # 1. Get per-sample predictions and targets
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["ids"]

            outputs = model(inputs, pair_indices, pair_mask)

            # Slice to scored length (68)
            outputs = outputs[:, : Config.PRED_LEN, :]
            targets = targets[:, : Config.PRED_LEN, :]

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
            all_ids.extend(ids)

    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)

    # 2. Calculate RMSE per sample (averaged over scored columns and positions)
    # Scored columns indices
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]

    # Filter columns
    y_pred_scored = y_pred[..., scored_indices]
    y_true_scored = y_true[..., scored_indices]

    # MSE per sample: Mean over (Length * Channels)
    mse_per_sample = torch.mean((y_true_scored - y_pred_scored) ** 2, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # 3. Load Metadata to correlate
    val_meta_df = pd.read_parquet(Config.VAL_METADATA_PATH)

    # Create a DataFrame for analysis
    analysis_df = pd.DataFrame({"id": all_ids, "error_rmse": rmse_per_sample})

    # Merge with metadata
    analysis_df = analysis_df.merge(val_meta_df, on="id", how="left")

    # Calculate correlations
    # We check numerical columns of interest
    features_to_check = ["signal_to_noise", "SN_filter"]
    # Also add derived features
    analysis_df["pct_A"] = analysis_df["sequence"].apply(
        lambda s: s.count("A") / len(s)
    )
    analysis_df["pct_unpaired"] = analysis_df["structure"].apply(
        lambda s: s.count(".") / len(s)
    )

    features_to_check.extend(["pct_A", "pct_unpaired"])

    print("Correlation of Error (RMSE) with features:")
    correlations = (
        analysis_df[features_to_check + ["error_rmse"]]
        .corr()["error_rmse"]
        .drop("error_rmse")
    )
    print(correlations)

    # =========================================================================
    # 7. Conditional Submission
    # =========================================================================
    THRESHOLD = 0.5884495377540588

    if final_val_score < THRESHOLD:
        print(
            f"\nValidation score ({final_val_score}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        ids_list = []
        preds_list = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["inputs"].to(device)
                pair_indices = batch["pair_indices"].to(device)
                pair_mask = batch["pair_mask"].to(device)
                ids = batch["ids"]

                outputs = model(inputs, pair_indices, pair_mask)
                outputs = outputs.cpu().numpy()

                ids_list.extend(ids)
                preds_list.append(outputs)

        all_preds = np.concatenate(preds_list, axis=0)

        submission_rows = []
        target_cols = Config.TARGET_COLS

        for i, sample_id in enumerate(ids_list):
            sample_preds = all_preds[i]  # Shape (107, 5)

            for seqpos in range(Config.SEQ_LEN):
                row_id = f"{sample_id}_{seqpos}"
                row_values = sample_preds[seqpos]

                row_dict = {"id_seqpos": row_id}
                for j, col in enumerate(target_cols):
                    row_dict[col] = float(row_values[j])

                submission_rows.append(row_dict)

        submission_df = pd.DataFrame(submission_rows)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation score ({final_val_score}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
