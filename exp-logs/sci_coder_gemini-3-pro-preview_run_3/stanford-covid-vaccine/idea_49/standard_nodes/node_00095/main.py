import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.optim as optim
import warnings

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data import get_dataloaders
from library.model import DeepStabilizedBiGRU
from library.loss import MCRMSELoss

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    # 1. Initialization
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fast Baseline Settings
    # Limiting epochs to ensure execution within 2 hours while maintaining sufficient convergence
    EPOCHS = 10

    # Ensure working directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # 2. Data Loading
    # Load cached data for speed
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Setup
    model = DeepStabilizedBiGRU().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    criterion = MCRMSELoss()

    # Identify indices of scored columns for metric calculation
    # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Config.SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    scored_indices = [Config.TARGET_COLS.index(col) for col in Config.SCORED_COLS]

    # 4. Training Loop
    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "baseline_best_model.pth")

    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        for batch in train_loader:
            x, bppm, y, _ = batch
            x = x.to(device)
            bppm = bppm.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            preds = model(x, bppm)
            loss = criterion(preds, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD)
            optimizer.step()

        # --- Validation ---
        model.eval()
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for batch in val_loader:
                x, bppm, y, _ = batch
                x = x.to(device)
                bppm = bppm.to(device)

                preds = model(x, bppm)

                # Store on CPU to save GPU memory
                val_preds_list.append(preds.cpu())
                val_targets_list.append(y.cpu())

        val_preds = torch.cat(val_preds_list, dim=0)
        val_targets = torch.cat(val_targets_list, dim=0)

        # Calculate metric on scored columns only
        val_mcrmse = calculate_metric(
            val_targets, val_preds, scored_cols_indices=scored_indices
        )

        # Save best model
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)

        scheduler.step()

    # 5. Final Evaluation & Metric Reporting
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Re-run validation to get IDs for failure analysis
    val_preds_list = []
    val_targets_list = []
    val_ids_list = []

    with torch.no_grad():
        for batch in val_loader:
            x, bppm, y, ids = batch
            x = x.to(device)
            bppm = bppm.to(device)

            preds = model(x, bppm)

            val_preds_list.append(preds.cpu())
            val_targets_list.append(y.cpu())
            val_ids_list.extend(ids)

    val_preds = torch.cat(val_preds_list, dim=0)
    val_targets = torch.cat(val_targets_list, dim=0)

    final_metric = calculate_metric(
        val_targets, val_preds, scored_cols_indices=scored_indices
    )
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    # Calculate RMSE per sample on scored columns/positions
    # Slice to scored length (68) and scored columns
    vp_sliced = val_preds[:, : Config.SEQ_SCORED, scored_indices]
    vt_sliced = val_targets[:, : Config.SEQ_SCORED, scored_indices]

    # MSE per sample: Mean over sequence (dim 1) and targets (dim 2)
    mse_per_sample = torch.mean((vp_sliced - vt_sliced) ** 2, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # Load metadata to correlate
    val_meta_df = pd.read_parquet(Config.VAL_FILE)

    # Create a DataFrame for errors
    error_df = pd.DataFrame({"id": val_ids_list, "rmse": rmse_per_sample})

    # Merge with metadata
    analysis_df = val_meta_df.merge(error_df, on="id")

    # Calculate simple features for correlation if not present
    if "pct_paired" not in analysis_df.columns:
        analysis_df["pct_paired"] = analysis_df["structure"].apply(
            lambda s: (s.count("(") + s.count(")")) / len(s)
        )

    # Select numeric columns for correlation
    corr_cols = ["rmse", "signal_to_noise", "SN_filter", "pct_paired"]
    # Filter to existing columns
    corr_cols = [c for c in corr_cols if c in analysis_df.columns]

    correlations = analysis_df[corr_cols].corr()["rmse"].drop("rmse")
    print("\nFailure Analysis - Correlation with Error (RMSE):")
    print(correlations)

    # 7. Submission Generation
    THRESHOLD = 0.5884495377540588

    if final_metric < THRESHOLD:
        test_preds_list = []
        test_ids_list = []

        with torch.no_grad():
            for batch in test_loader:
                x, bppm, _, ids = batch
                x = x.to(device)
                bppm = bppm.to(device)

                # Predict (N, 107, 5)
                preds = model(x, bppm)

                test_preds_list.append(preds.cpu().numpy())
                test_ids_list.extend(ids)

        all_preds = np.concatenate(test_preds_list, axis=0)

        # Format submission
        submission_data = []
        target_cols = Config.TARGET_COLS

        for i, sample_id in enumerate(test_ids_list):
            sample_preds = all_preds[i]  # (107, 5)

            for seqpos in range(Config.SEQ_LEN):
                row_id = f"{sample_id}_{seqpos}"
                row_vals = sample_preds[seqpos]

                row_dict = {"id_seqpos": row_id}
                for t_idx, t_name in enumerate(target_cols):
                    row_dict[t_name] = row_vals[t_idx]

                submission_data.append(row_dict)

        submission_df = pd.DataFrame(submission_data)
        submission_path = "./submission/submission.csv"
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    main()
