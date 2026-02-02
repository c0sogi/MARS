import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import library modules
from library.config import Config
from library.model import DeepBiasRefinedBiGRU
from library.data import get_dataloaders
from library.loss import MCRMSELoss
from library.utils import set_seed, calculate_metric


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Override Config for fast baseline execution
    EPOCHS = 15

    # 2. Data Loading
    # We use the full dataset but fewer epochs for the fast baseline.
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = DeepBiasRefinedBiGRU().to(device)

    # 4. Training Setup
    criterion = MCRMSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=EPOCHS * len(train_loader), eta_min=Config.MIN_LR
    )

    # 5. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        for batch in train_loader:
            # Move data to device
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(features, pair_indices)

            # Loss calculation
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient Clipping (Critical for stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD_NORM)

            optimizer.step()
            scheduler.step()

    # 6. Validation
    model.eval()
    val_preds = []
    val_targets = []
    val_ids = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            targets = batch["targets"]  # Keep on CPU for accumulation
            ids = batch["ids"]

            outputs = model(features, pair_indices)

            val_preds.append(outputs.cpu().numpy())
            val_targets.append(targets.numpy())
            val_ids.extend(ids)

    # Concatenate all batches
    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Calculate Metric
    final_metric = calculate_metric(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("\n==== Failure Analysis ====")
    # Load validation metadata to get auxiliary features
    val_meta_path = os.path.join(Config.METADATA_DIR, "val.parquet")
    if os.path.exists(val_meta_path):
        val_df = pd.read_parquet(val_meta_path)

        # Ensure alignment (val_loader is not shuffled, so order should match if index wasn't reset weirdly)
        # We map errors back to IDs

        # Calculate RMSE per sample (on scored columns only)
        # Slice to scored length
        y_true_scored = val_targets[:, : Config.PRED_LEN, :]
        y_pred_scored = val_preds[:, : Config.PRED_LEN, :]

        # Filter for scored columns indices
        scored_indices = [
            i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
        ]

        y_true_s = y_true_scored[:, :, scored_indices]
        y_pred_s = y_pred_scored[:, :, scored_indices]

        # MSE per sample: (N, L, 3) -> mean over L, 3 -> (N,)
        mse_per_sample = np.mean((y_true_s - y_pred_s) ** 2, axis=(1, 2))
        rmse_per_sample = np.sqrt(mse_per_sample)

        # Create DataFrame for analysis
        error_df = pd.DataFrame({"id": val_ids, "error": rmse_per_sample})

        # Merge with metadata
        analysis_df = pd.merge(error_df, val_df, on="id", how="left")

        # Calculate correlations
        # Select numerical columns of interest
        cols_to_corr = ["signal_to_noise", "SN_filter", "seq_length"]
        # Add nucleotide counts if available or compute them
        if "sequence" in analysis_df.columns:
            analysis_df["pct_A"] = analysis_df["sequence"].apply(
                lambda x: x.count("A") / len(x)
            )
            analysis_df["pct_U"] = analysis_df["sequence"].apply(
                lambda x: x.count("U") / len(x)
            )
            analysis_df["pct_G"] = analysis_df["sequence"].apply(
                lambda x: x.count("G") / len(x)
            )
            analysis_df["pct_C"] = analysis_df["sequence"].apply(
                lambda x: x.count("C") / len(x)
            )
            cols_to_corr.extend(["pct_A", "pct_U", "pct_G", "pct_C"])

        correlations = (
            analysis_df[cols_to_corr + ["error"]].corr()["error"].drop("error")
        )
        print("Correlation between Error and Features:")
        print(correlations)
    else:
        print("Validation metadata not found. Skipping failure analysis.")

    # 8. Submission
    THRESHOLD = 0.5884495377540588
    if final_metric < THRESHOLD:
        print("\nMetric below threshold. Generating submission...")

        test_preds = []
        test_ids = []

        with torch.no_grad():
            for batch in test_loader:
                features = batch["features"].to(device)
                pair_indices = batch["pair_indices"].to(device)
                ids = batch["ids"]

                outputs = model(features, pair_indices)

                test_preds.append(outputs.cpu().numpy())
                test_ids.extend(ids)

        test_preds = np.concatenate(test_preds, axis=0)  # (N_test, 107, 5)

        # Prepare submission DataFrame
        # We need 107 rows per sample
        submission_rows = []

        target_cols = (
            Config.TARGET_COLS
        )  # ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

        for i, sample_id in enumerate(test_ids):
            sample_pred = test_preds[i]  # (107, 5)

            for seqpos in range(Config.SEQ_LEN):
                row_id = f"{sample_id}_{seqpos}"
                row_data = {
                    "id_seqpos": row_id,
                    "reactivity": sample_pred[seqpos, 0],
                    "deg_Mg_pH10": sample_pred[seqpos, 1],
                    "deg_pH10": sample_pred[seqpos, 2],
                    "deg_Mg_50C": sample_pred[seqpos, 3],
                    "deg_50C": sample_pred[seqpos, 4],
                }
                submission_rows.append(row_data)

        submission_df = pd.DataFrame(submission_rows)

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
