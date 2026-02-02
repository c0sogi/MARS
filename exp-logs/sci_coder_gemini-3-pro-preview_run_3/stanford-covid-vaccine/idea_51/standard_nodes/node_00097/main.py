import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.data_utils import get_dataloaders, seed_everything
from library.model import DeepStabilizedBiGRU
from library.loss_metrics import MCRMSELoss, calculate_metric_mcrmse
from library.train_eval import train_epoch, validate, generate_submission


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates sample-wise error and correlates with metadata features.
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
            bpp_indices = batch["bpp_indices"].to(device)
            bpp_masks = batch["bpp_masks"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["ids"]

            outputs = model(inputs, bpp_indices, bpp_masks)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_ids.extend(ids)

    preds = np.concatenate(all_preds, axis=0)  # (N, 107, 5)
    targets = np.concatenate(all_targets, axis=0)  # (N, 107, 5)
    ids = np.array(all_ids)

    # 2. Calculate Error per Sample
    # Filter for scored columns and positions
    # Scored cols: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]
    pred_len = Config.PRED_LEN  # 68

    preds_sliced = preds[:, :pred_len, scored_indices]
    targets_sliced = targets[:, :pred_len, scored_indices]

    # MSE per sample (average over positions and columns)
    mse_per_sample = np.mean((preds_sliced - targets_sliced) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # 3. Load Metadata
    val_meta_path = Config.VAL_PATH
    if not os.path.exists(val_meta_path):
        print("Validation metadata not found. Skipping detailed correlation analysis.")
        return

    val_df = pd.read_parquet(val_meta_path)

    # Create a DataFrame for analysis
    analysis_df = pd.DataFrame({"id": ids, "error_rmse": rmse_per_sample})

    # Merge with metadata
    merged_df = pd.merge(analysis_df, val_df, on="id", how="left")

    # 4. Compute Correlations
    features_to_check = ["signal_to_noise", "SN_filter", "seq_length"]
    # Add nucleotide content
    merged_df["pct_A"] = merged_df["sequence"].apply(lambda s: s.count("A") / len(s))
    merged_df["pct_U"] = merged_df["sequence"].apply(lambda s: s.count("U") / len(s))
    merged_df["pct_G"] = merged_df["sequence"].apply(lambda s: s.count("G") / len(s))
    merged_df["pct_C"] = merged_df["sequence"].apply(lambda s: s.count("C") / len(s))

    features_to_check.extend(["pct_A", "pct_U", "pct_G", "pct_C"])

    print(f"{'Feature':<20} | {'Correlation (r)':<15}")
    print("-" * 40)

    for feat in features_to_check:
        if feat in merged_df.columns:
            # Drop NaNs just in case
            valid_data = merged_df[[feat, "error_rmse"]].dropna()
            if len(valid_data) > 1:
                # Ensure numeric
                try:
                    x = valid_data[feat].astype(float)
                    y = valid_data["error_rmse"].astype(float)
                    corr, _ = pearsonr(x, y)
                    print(f"{feat:<20} | {corr:.4f}")
                except Exception:
                    pass


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = DeepStabilizedBiGRU().to(device)

    # 4. Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )
    criterion = MCRMSELoss()

    # 5. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")
    best_metric = float("inf")

    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metric = validate(model, val_loader, device)
        scheduler.step()

        # Checkpointing
        if val_metric < best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), Config.MODEL_PATH)
            # print(f"Epoch {epoch+1}: New best model (MCRMSE: {best_metric:.6f})")

        # Optional: Print progress every 5 epochs to keep log clean
        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val MCRMSE: {val_metric:.5f}"
            )

    # 6. Final Validation & Reporting
    print("Training complete. Loading best model...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    final_metric = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 8. Submission
    THRESHOLD = 0.5884495377540588
    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} meets threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"\nMetric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
