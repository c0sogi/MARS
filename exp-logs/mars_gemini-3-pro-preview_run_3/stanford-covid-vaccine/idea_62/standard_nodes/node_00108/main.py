import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from library
from library.config import Config
from library.data import get_dataloaders, RNADataset
from library.model import RNAModel
from library.engine import fit, set_seed, validate
from library.utils import calculate_metric, get_scored_col_indices


def run_failure_analysis(val_preds, val_targets, val_ids):
    """
    Analyzes model performance by correlating error with input features.
    """
    print("\n==== Failure Analysis ====")

    # 1. Calculate RMSE per sample
    # val_preds: (N, 107, 5), val_targets: (N, 68, 5)
    # Slice preds to 68
    seq_scored = Config.SEQ_SCORED
    preds_sliced = val_preds[:, :seq_scored, :]

    # Filter for scored columns
    scored_indices = get_scored_col_indices()
    preds_scored = preds_sliced[:, :, scored_indices]
    targets_scored = val_targets[:, :, scored_indices]

    # Calculate RMSE per sample (averaged over positions and columns)
    # Shape: (N,)
    mse_per_sample = np.mean((preds_scored - targets_scored) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    error_df = pd.DataFrame({"id": val_ids, "rmse": rmse_per_sample})

    # 2. Load Metadata
    if os.path.exists(Config.VAL_METADATA_PATH):
        meta_df = pd.read_parquet(Config.VAL_METADATA_PATH)

        # Merge
        analysis_df = pd.merge(error_df, meta_df, on="id", how="inner")

        # Feature Engineering for correlation
        analysis_df["pct_A"] = analysis_df["sequence"].apply(
            lambda s: s.count("A") / len(s)
        )
        analysis_df["pct_G"] = analysis_df["sequence"].apply(
            lambda s: s.count("G") / len(s)
        )
        analysis_df["pct_U"] = analysis_df["sequence"].apply(
            lambda s: s.count("U") / len(s)
        )
        analysis_df["pct_C"] = analysis_df["sequence"].apply(
            lambda s: s.count("C") / len(s)
        )
        analysis_df["pct_unpaired"] = analysis_df["structure"].apply(
            lambda s: s.count(".") / len(s)
        )

        # Select numerical columns for correlation
        corr_cols = [
            "rmse",
            "signal_to_noise",
            "SN_filter",
            "pct_A",
            "pct_G",
            "pct_U",
            "pct_C",
            "pct_unpaired",
        ]
        # Ensure columns exist
        corr_cols = [c for c in corr_cols if c in analysis_df.columns]

        correlations = (
            analysis_df[corr_cols].corr()["rmse"].sort_values(ascending=False)
        )
        print("Correlation between Error (RMSE) and Features:")
        print(correlations)
    else:
        print("Validation metadata not found. Skipping detailed failure analysis.")


def generate_submission(model, test_loader, device):
    """
    Generates submission.csv for the test set.
    """
    print("\nGenerating submission...")
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            bpp_indices = batch["bpp_indices"].to(device)
            bpp_masks = batch["bpp_masks"].to(device)
            ids = batch["id"]

            outputs = model(inputs, bpp_indices, bpp_masks)
            all_preds.append(outputs.cpu().numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)  # (N, 107, 5)

    # Format for submission
    # Need rows: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_rows = []
    target_cols = Config.TARGET_COLS

    for i, sample_id in enumerate(all_ids):
        sample_preds = all_preds[i]  # (107, 5)
        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos].tolist()

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = row_values[col_idx]

            submission_rows.append(row_dict)

    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    # 1. Setup
    # Override epochs for fast baseline
    Config.EPOCHS = 15

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing model...")
    model = RNAModel().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR)

    # 5. Training
    print("Starting training...")
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
        patience=Config.PATIENCE,
        save_path=Config.MODEL_SAVE_PATH,
    )

    # 6. Evaluation
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Get validation predictions for metric and analysis
    val_preds = []
    val_targets = []
    val_ids = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            bpp_indices = batch["bpp_indices"].to(device)
            bpp_masks = batch["bpp_masks"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            outputs = model(inputs, bpp_indices, bpp_masks)

            val_preds.append(outputs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())
            val_ids.extend(ids)

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Calculate Metric
    final_metric = calculate_metric(val_preds, val_targets)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    run_failure_analysis(val_preds, val_targets, val_ids)

    # 8. Submission
    THRESHOLD = 0.5884495377540588
    if final_metric < THRESHOLD:
        generate_submission(model, test_loader, device)
    else:
        print(
            f"Validation metric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
