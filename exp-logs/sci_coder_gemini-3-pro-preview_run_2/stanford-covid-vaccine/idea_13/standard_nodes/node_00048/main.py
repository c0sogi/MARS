import os
import sys
import numpy as np
import pandas as pd
import torch
import scipy.stats as stats

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, calculate_global_mcrmse
from library.train import Trainer
from library.data import get_dataloaders
from library.model import BridgedHybridNet


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for fast execution and isolation
    Config.EPOCHS = 20  # Limit epochs for a fast baseline (default was 50)
    Config.WORKING_DIR = "./working/idea_14/"  # Isolate working directory

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set reproducibility
    set_seed(Config.SEED)

    print(f"Configuration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Working Dir: {Config.WORKING_DIR}")

    # ==========================================
    # 2. Training Pipeline
    # ==========================================
    print("\nInitializing Trainer...")
    trainer = Trainer()

    print("Starting Training...")
    trainer.run_training()

    # ==========================================
    # 3. Final Validation Evaluation
    # ==========================================
    print("\nPerforming Final Validation...")

    # Load the best model saved during training
    model_path = Config.get_model_path()
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    trainer.model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    trainer.model.eval()

    # Get Validation Loader
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    all_preds = []
    all_targets = []
    all_ids = []

    # Inference loop (No Grad)
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(Config.DEVICE)
            partner_indices = batch["partner_indices"].to(Config.DEVICE)
            # Targets: (Batch, 5, Seq_Len) -> Permute to (Batch, Seq_Len, 5)
            targets = batch["targets"].to(Config.DEVICE).permute(0, 2, 1)
            ids = batch["ids"]

            preds = trainer.model(inputs, partner_indices)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate results
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Filter for Scored Columns and Scored Sequence Length
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]

    # Slice: (N_Samples, Seq_Scored, N_Scored_Cols)
    preds_scored = all_preds[:, : Config.SEQ_SCORED, scored_indices]
    targets_scored = all_targets[:, : Config.SEQ_SCORED, scored_indices]

    # Calculate Metric
    final_metric = calculate_global_mcrmse(preds_scored, targets_scored)
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\nPerforming Failure Analysis...")

    # Calculate RMSE per sample (averaged over scored positions and columns)
    # Squared Error
    se = (preds_scored - targets_scored) ** 2
    # Mean Squared Error per sample
    mse_per_sample = np.mean(se, axis=(1, 2))
    # RMSE per sample
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Load Validation Metadata to get features
    val_df = pd.read_csv(Config.VAL_CSV)

    # Ensure alignment between predictions and metadata
    val_df.set_index("id", inplace=True)
    # Reindex val_df to match the order of all_ids processed in the loader
    val_df_ordered = val_df.loc[all_ids]

    # Construct Analysis DataFrame
    analysis_df = pd.DataFrame(
        {
            "error": rmse_per_sample,
            "signal_to_noise": val_df_ordered["signal_to_noise"].values,
            "SN_filter": val_df_ordered["SN_filter"].values,
            "mean_reactivity": val_df_ordered["mean_reactivity"].values,
        }
    )

    # Add sequence composition features
    sequences = val_df_ordered["sequence"].values
    analysis_df["count_A"] = [s.count("A") for s in sequences]
    analysis_df["count_G"] = [s.count("G") for s in sequences]
    analysis_df["count_C"] = [s.count("C") for s in sequences]
    analysis_df["count_U"] = [s.count("U") for s in sequences]

    # Calculate Correlations
    correlations = analysis_df.corr()["error"].sort_values(ascending=False)

    print("Correlation between Model Error and Input Features:")
    print(correlations)

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    THRESHOLD = 0.5421870350837708

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} is better than threshold {THRESHOLD}. Generating submission..."
        )
        trainer.generate_submission()
    else:
        print(
            f"\nMetric {final_metric} did not beat threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
