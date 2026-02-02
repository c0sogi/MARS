import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, MCRMSELoss, compute_val_metric
from library.data import get_loaders, load_data_cached
from library.model import RNAModel
from library.engine import train_fn, inference_fn


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override Config for Fast Baseline
    Config.epochs = 15  # Reduced from 50 for speed
    Config.batch_size = 32

    # Set seeds for reproducibility
    set_seed(Config.seed)

    device = torch.device(Config.device)
    print(f"Running on device: {device}")

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    print("Loading datasets...")
    # We use load_cached_data=True to leverage pre-processing
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=Config.batch_size,
        num_workers=Config.num_workers,
        load_cached_data=True,
    )

    # =========================================================================
    # 3. Model Initialization
    # =========================================================================
    print("Initializing High-Capacity GLU-Refined BiGRU Model...")
    model = RNAModel(config=Config)
    model.to(device)

    # =========================================================================
    # 4. Optimization Setup
    # =========================================================================
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=Config.eta_min
    )

    criterion = MCRMSELoss()

    # =========================================================================
    # 5. Training Loop
    # =========================================================================
    best_score = float("inf")
    best_model_path = Config.MODEL_PATH

    print(f"Starting training for {Config.epochs} epochs...")

    for epoch in range(Config.epochs):
        # Train Step
        train_loss = train_fn(model, train_loader, criterion, optimizer, device)

        # Validation Step (Manual eval to keep predictions for logic if needed,
        # but here we just need the score for model selection)
        model.eval()
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for x, pair_indices, pair_mask, y in val_loader:
                x = x.to(device)
                pair_indices = pair_indices.to(device)
                pair_mask = pair_mask.to(device)

                preds = model(x, pair_indices, pair_mask)
                val_preds_list.append(preds.cpu().numpy())
                val_targets_list.append(y.numpy())

        val_preds_epoch = np.concatenate(val_preds_list, axis=0)
        val_targets_epoch = np.concatenate(val_targets_list, axis=0)

        val_score = compute_val_metric(val_preds_epoch, val_targets_epoch)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{Config.epochs} | Train Loss: {train_loss:.5f} | Val MCRMSE: {val_score:.5f} | LR: {current_lr:.2e}"
        )

        # Save Best Model
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            # print(f"  -> New best model saved.")

    # =========================================================================
    # 6. Final Validation & Metric
    # =========================================================================
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Run inference on full validation set
    val_preds_list = []
    val_targets_list = []

    with torch.no_grad():
        for x, pair_indices, pair_mask, y in val_loader:
            x = x.to(device)
            pair_indices = pair_indices.to(device)
            pair_mask = pair_mask.to(device)
            preds = model(x, pair_indices, pair_mask)
            val_preds_list.append(preds.cpu().numpy())
            val_targets_list.append(y.numpy())

    val_preds = np.concatenate(val_preds_list, axis=0)
    val_targets = np.concatenate(val_targets_list, axis=0)

    final_metric = compute_val_metric(val_preds, val_targets)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # =========================================================================
    # 7. Failure Analysis
    # =========================================================================
    print("\nRunning Failure Analysis...")

    # 1. Calculate Sample-wise Error (MCRMSE per sample)
    # Filter for scored columns [0, 1, 3] corresponding to reactivity, deg_Mg_pH10, deg_Mg_50C
    scored_indices = [0, 1, 3]

    # Slice to seq_scored (68)
    if val_preds.shape[1] > Config.seq_scored:
        vp = val_preds[:, : Config.seq_scored, :][:, :, scored_indices]
        vt = val_targets[:, :, scored_indices]
    else:
        vp = val_preds[:, :, scored_indices]
        vt = val_targets[:, :, scored_indices]

    # MSE per sample: (N, 68, 3) -> mean over (68, 3) is not quite right for MCRMSE.
    # MCRMSE definition: Average of RMSEs of columns.
    # We want to see which samples contribute most to error.
    # Let's define Sample Error as the mean of the RMSEs of the 3 columns for that sample.
    # (N, 68, 3) -> (N, 3) RMSE per column per sample -> (N,) Mean RMSE
    sq_diff = (vp - vt) ** 2
    rmse_per_col_per_sample = np.sqrt(np.mean(sq_diff, axis=1))  # Shape (N, 3)
    sample_errors = np.mean(rmse_per_col_per_sample, axis=1)  # Shape (N,)

    # 2. Load Metadata to correlate
    val_df = pd.read_parquet(Config.VAL_METADATA)

    # Ensure alignment (val_loader is shuffle=False, so order matches df)
    if len(val_df) != len(sample_errors):
        print(
            f"Warning: Validation DF length ({len(val_df)}) != Predictions length ({len(sample_errors)})"
        )
        # Truncate to minimum
        min_len = min(len(val_df), len(sample_errors))
        val_df = val_df.iloc[:min_len]
        sample_errors = sample_errors[:min_len]

    # Extract features for correlation
    analysis_df = pd.DataFrame()
    analysis_df["error"] = sample_errors

    # Metadata features
    if "signal_to_noise" in val_df.columns:
        analysis_df["signal_to_noise"] = val_df["signal_to_noise"].values
    if "SN_filter" in val_df.columns:
        analysis_df["SN_filter"] = val_df["SN_filter"].values

    # Sequence features
    analysis_df["pct_unpaired"] = (
        val_df["structure"].apply(lambda x: x.count(".") / len(x)).values
    )
    analysis_df["pct_A"] = (
        val_df["sequence"].apply(lambda x: x.count("A") / len(x)).values
    )
    analysis_df["pct_G"] = (
        val_df["sequence"].apply(lambda x: x.count("G") / len(x)).values
    )
    analysis_df["pct_U"] = (
        val_df["sequence"].apply(lambda x: x.count("U") / len(x)).values
    )

    # Compute Correlations
    correlations = analysis_df.corr()["error"].sort_values(ascending=False)
    print("Correlation between Model Error and Input Features:")
    print(correlations.drop("error"))

    # =========================================================================
    # 8. Submission Generation
    # =========================================================================
    THRESHOLD = 0.5884495377540588

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")

        # Create output directory
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        # Inference on Test Set
        test_preds, test_ids = inference_fn(model, test_loader, device)

        # Format Data
        submission_rows = []
        target_cols = Config.target_cols

        for i, sample_id in enumerate(test_ids):
            # Predictions shape: (107, 5)
            sample_pred = test_preds[i]

            for seqpos in range(Config.seq_length):
                row_id = f"{sample_id}_{seqpos}"
                row_vals = sample_pred[seqpos]

                row_dict = {"id_seqpos": row_id}
                for col_idx, col_name in enumerate(target_cols):
                    row_dict[col_name] = float(row_vals[col_idx])

                submission_rows.append(row_dict)

        submission_df = pd.DataFrame(submission_rows)

        # Reorder columns
        cols = ["id_seqpos"] + target_cols
        submission_df = submission_df[cols]

        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nMetric {final_metric} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
