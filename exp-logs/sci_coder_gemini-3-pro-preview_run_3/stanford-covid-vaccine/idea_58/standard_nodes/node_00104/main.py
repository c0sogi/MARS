import sys
import os
import torch
import pandas as pd
import numpy as np
import warnings

# Ensure library imports work by adding current directory to path
sys.path.append(os.getcwd())

from library.config import Config
from library.train_eval import run_training, generate_submission, set_seed
from library.dataset import RNADataset
from library.model import HCSDBiGRU
from library.loss_metric import compute_competition_metric
from torch.utils.data import DataLoader

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # 1. Run Training
    # We limit epochs to 15 for a fast baseline execution.
    # The dataset is small (1728 samples), so this will be very quick on GPU.
    print("Starting Training...")
    run_training(epochs=15)

    # 2. Validation Inference
    print("Running Validation Inference...")
    device = torch.device(Config.DEVICE)

    # Load the best model
    model = HCSDBiGRU().to(device)
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load Validation Dataset
    val_dataset = RNADataset(mode="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    all_preds = []
    all_targets = []
    all_ids = []

    # Inference loop
    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            bpp_indices = batch["bpp_indices"].to(device)
            bpp_mask = batch["bpp_mask"].to(device)
            targets = batch["targets"]  # Keep targets on CPU
            ids = batch["ids"]

            outputs = model(features, bpp_indices, bpp_mask)

            all_preds.append(outputs.cpu())
            all_targets.append(targets)
            all_ids.extend(ids)

    # Concatenate results
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # 3. Compute and Print Metric
    metric = compute_competition_metric(all_preds, all_targets)
    print(f"Final Validation Metric: {metric}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate RMSE per sample (Error Magnitude)
    # We focus on the 3 scored columns: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
    scored_indices = [0, 1, 3]
    seq_scored = Config.SEQ_SCORED

    preds_scored = all_preds[:, :seq_scored, scored_indices]
    targets_scored = all_targets[:, :seq_scored, scored_indices]

    # MSE per sample (averaged over positions and columns)
    mse_per_sample = torch.mean((preds_scored - targets_scored) ** 2, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # Load metadata to get features
    val_df = pd.read_parquet(Config.VAL_PATH)
    val_df.set_index("id", inplace=True)

    analysis_data = []

    for i, sample_id in enumerate(all_ids):
        if sample_id not in val_df.index:
            continue

        row = val_df.loc[sample_id]

        # Extract features
        sn_ratio = row.get("signal_to_noise", 0)
        sn_filter = row.get("SN_filter", 0)
        seq = row.get("sequence", "")

        # Sequence composition
        seq_len = len(seq) if len(seq) > 0 else 1
        pct_A = seq.count("A") / seq_len
        pct_G = seq.count("G") / seq_len
        pct_C = seq.count("C") / seq_len
        pct_U = seq.count("U") / seq_len

        analysis_data.append(
            {
                "error": rmse_per_sample[i],
                "signal_to_noise": sn_ratio,
                "SN_filter": sn_filter,
                "pct_A": pct_A,
                "pct_G": pct_G,
                "pct_C": pct_C,
                "pct_U": pct_U,
            }
        )

    # Create DataFrame and compute correlations
    if analysis_data:
        analysis_df = pd.DataFrame(analysis_data)
        correlations = analysis_df.corr()["error"].sort_values(ascending=False)
        print("Correlation between Error and Features:")
        print(correlations)
    else:
        print("Could not perform failure analysis: No matching IDs found.")

    # 5. Generate Submission
    # Threshold defined in the task
    THRESHOLD = 0.5884495377540588

    if metric < THRESHOLD:
        print(
            f"\nValidation metric {metric} is lower than {THRESHOLD}. Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"\nValidation metric {metric} is NOT lower than {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
