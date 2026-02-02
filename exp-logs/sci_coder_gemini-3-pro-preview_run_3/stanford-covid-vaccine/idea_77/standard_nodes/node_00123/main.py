import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.train import run_training, generate_submission
from library.data import get_dataloaders
from library.model import RNAModel
from library.utils import seed_everything, MCRMSE


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    print("Starting pipeline...")

    # 2. Training
    # We limit max_epochs to 10 for a fast baseline execution.
    # The dataset size is small (~1.7k), so full data usage is efficient.
    best_model_path = run_training(max_epochs=10, batch_size=Config.BATCH_SIZE)

    # 3. Validation Inference
    print("Loading best model for validation...")
    device = Config.DEVICE
    model = RNAModel(Config).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Load validation data
    _, val_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    all_preds = []
    all_targets = []

    print("Generating validation predictions...")
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            targets = batch["targets"].to(device)

            preds = model(inputs, pair_indices, pair_mask)

            # Move to CPU for metric calculation
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # 4. Metric Calculation
    # Scored columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    # Scored length: 68
    scorer = MCRMSE(pred_len=Config.PRED_LEN, scored_indices=[0, 1, 3])
    final_metric = scorer(all_preds, all_targets).item()

    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate RMSE per sample for the scored columns/length
    p_slice = all_preds[:, : Config.PRED_LEN, :]
    t_slice = all_targets[:, : Config.PRED_LEN, :]

    scored_cols = [0, 1, 3]
    p_filtered = p_slice[:, :, scored_cols]
    t_filtered = t_slice[:, :, scored_cols]

    # Mean over length and columns, then sqrt -> RMSE per sample
    mse_per_sample = torch.mean((p_filtered - t_filtered) ** 2, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # Load metadata
    if os.path.exists(Config.VAL_PATH):
        val_df = pd.read_parquet(Config.VAL_PATH)

        # Create analysis dataframe
        analysis_df = val_df.copy()
        analysis_df["error_rmse"] = rmse_per_sample

        # Feature Engineering: Nucleotide content
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

        # Calculate correlations
        features = ["signal_to_noise", "pct_A", "pct_G", "pct_C", "pct_U"]
        correlations = analysis_df[features].corrwith(analysis_df["error_rmse"])

        print("Correlation between Model Error (RMSE) and Input Features:")
        print(correlations)
    else:
        print("Validation metadata not found. Skipping detailed failure analysis.")

    # 6. Submission
    THRESHOLD = 0.5884495377540588
    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) < Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(best_model_path)
    else:
        print(
            f"\nMetric ({final_metric}) >= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
