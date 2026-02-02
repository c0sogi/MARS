import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data import get_dataloaders, RNADataset
from library.model import DeepResGLUBiGRU
from library.train import Trainer


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline
    # Reducing epochs to ensure completion within strict time limits while allowing convergence
    Config.EPOCHS = 15

    # Ensure output directories exist
    os.makedirs("./submission", exist_ok=True)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading data...")
    # Load cached data if available to speed up startup
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # -------------------------------------------------------------------------
    # 3. Model Training
    # -------------------------------------------------------------------------
    print("Initializing model...")
    model = DeepResGLUBiGRU()

    print("Starting training...")
    trainer = Trainer(model, train_loader, val_loader, Config)
    trainer.fit()

    # -------------------------------------------------------------------------
    # 4. Validation & Metric Calculation
    # -------------------------------------------------------------------------
    print("Performing final validation...")

    # Load the best model saved during training
    best_model_path = Config.MODEL_SAVE_PATH
    if not os.path.exists(best_model_path):
        print("Error: Best model file not found.")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    val_preds = []
    val_targets = []
    val_ids = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            targets = batch["targets"]  # Keep on CPU
            ids = batch["id"]

            outputs = model(inputs, pair_indices)

            val_preds.append(outputs.cpu().numpy())
            val_targets.append(targets.numpy())
            val_ids.extend(ids)

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Calculate MCRMSE
    final_metric = calculate_metric(val_preds, val_targets)
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nRunning Failure Analysis...")

    # Calculate error per sample (RMSE across scored positions and scored columns)
    # Target indices: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = Config.SCORED_TARGET_INDICES
    seq_scored = Config.SEQ_SCORED

    # Slice to scored region
    preds_sliced = val_preds[:, :seq_scored, :][:, :, scored_indices]
    targets_sliced = val_targets[:, :, scored_indices]

    # MSE per sample: (N, 68, 3) -> (N,)
    mse_per_sample = np.mean((preds_sliced - targets_sliced) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Create Analysis DataFrame
    analysis_df = pd.DataFrame({"id": val_ids, "error": rmse_per_sample})

    # Load Metadata to get features
    val_meta_path = Config.VAL_METADATA
    val_df = pd.read_parquet(val_meta_path)

    # Merge error with metadata
    analysis_df = analysis_df.merge(val_df, on="id", how="left")

    # Feature Engineering for Correlation
    analysis_df["pct_A"] = analysis_df["sequence"].apply(
        lambda s: s.count("A") / len(s)
    )
    analysis_df["pct_G"] = analysis_df["sequence"].apply(
        lambda s: s.count("G") / len(s)
    )
    analysis_df["pct_C"] = analysis_df["sequence"].apply(
        lambda s: s.count("C") / len(s)
    )
    analysis_df["pct_U"] = analysis_df["sequence"].apply(
        lambda s: s.count("U") / len(s)
    )

    # Calculate Correlations
    correlate_cols = ["signal_to_noise", "pct_A", "pct_G", "pct_C", "pct_U"]
    print("Correlation between Error and Features:")
    for col in correlate_cols:
        if col in analysis_df.columns:
            corr = analysis_df["error"].corr(analysis_df[col])
            print(f"  {col}: {corr:.4f}")

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 0.5884495377540588

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) < Threshold ({THRESHOLD}). Generating submission..."
        )

        test_preds = []
        test_ids = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["inputs"].to(device)
                pair_indices = batch["pair_indices"].to(device)
                ids = batch["id"]

                outputs = model(inputs, pair_indices)
                test_preds.append(outputs.cpu().numpy())
                test_ids.extend(ids)

        test_preds = np.concatenate(test_preds, axis=0)  # (N, 107, 5)

        # Format for submission
        # Need rows: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        submission_data = []
        target_cols = (
            Config.TARGET_COLS
        )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

        for i, sample_id in enumerate(test_ids):
            sample_pred = test_preds[i]  # (107, 5)
            for seqpos in range(Config.SEQ_LEN):
                row_id = f"{sample_id}_{seqpos}"
                row_values = sample_pred[seqpos].tolist()
                submission_data.append([row_id] + row_values)

        columns = ["id_seqpos"] + target_cols
        submission_df = pd.DataFrame(submission_data, columns=columns)

        save_path = "./submission/submission.csv"
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nMetric ({final_metric}) >= Threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
