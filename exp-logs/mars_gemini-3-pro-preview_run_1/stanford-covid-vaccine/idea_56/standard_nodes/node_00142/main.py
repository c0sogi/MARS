import pandas as pd
import numpy as np
import torch
import scipy.stats as stats
import os

from library.config import Config
from library.utils import set_seed, compute_mcrmse
from library.data import get_dataloaders
from library.model import RNAModel
from library.train import Trainer


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing model...")
    model = RNAModel().to(device)

    # 4. Training
    # The Trainer handles the training loop, validation per epoch, and saving the best model.
    trainer = Trainer(model, device, train_loader, val_loader, test_loader)
    trainer.fit()

    # 5. Final Evaluation & Failure Analysis
    print("\nRunning Final Evaluation on Best Model...")

    # Load the best checkpoint saved by the trainer
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model checkpoint not found. Using current state.")

    model.eval()

    all_preds = []
    all_targets = []
    all_ids = []

    # Inference on validation set
    with torch.no_grad():
        for batch in val_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            preds = model(seq, loop, dist)

            # Slice to scored length (68) for accurate metric calculation
            preds_sliced = preds[:, : Config.PRED_LEN, :]
            targets_sliced = targets[:, : Config.PRED_LEN, :]

            all_preds.append(preds_sliced.cpu().numpy())
            all_targets.append(targets_sliced.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate results
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute and Print Final Metric
    val_mcrmse = compute_mcrmse(all_preds, all_targets)
    print(f"Final Validation Metric: {val_mcrmse}")

    # --- Failure Analysis ---
    print("\n=== Failure Analysis ===")

    # Calculate RMSE per sample (averaging over positions and channels)
    # Shape: (N_samples, 68, 3) -> (N_samples,)
    mse_per_sample = np.mean((all_preds - all_targets) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Create a mapping from ID to Error
    error_map = dict(zip(all_ids, rmse_per_sample))

    # Load validation metadata to get features
    if os.path.exists(Config.VAL_FILE):
        df_val = pd.read_parquet(Config.VAL_FILE)

        # Add model error to dataframe
        df_val["model_error"] = df_val["id"].map(error_map)

        # Derive sequence features
        df_val["len_A"] = df_val["sequence"].apply(lambda x: x.count("A"))
        df_val["len_G"] = df_val["sequence"].apply(lambda x: x.count("G"))
        df_val["len_C"] = df_val["sequence"].apply(lambda x: x.count("C"))
        df_val["len_U"] = df_val["sequence"].apply(lambda x: x.count("U"))

        # Features to correlate
        features_to_analyze = [
            "signal_to_noise",
            "SN_filter",
            "len_A",
            "len_G",
            "len_C",
            "len_U",
        ]

        print("Correlation between Model Error (RMSE) and Input Features:")
        for feat in features_to_analyze:
            if feat in df_val.columns:
                # Ensure no NaNs in comparison
                subset = df_val[[feat, "model_error"]].dropna()
                if len(subset) > 0:
                    corr, _ = stats.pearsonr(subset[feat], subset["model_error"])
                    print(f"  {feat}: {corr:.4f}")
    else:
        print("Validation metadata file not found. Skipping detailed failure analysis.")

    # 6. Submission
    threshold = 0.6176461577
    if val_mcrmse < threshold:
        print(f"\nValidation metric {val_mcrmse} is better than threshold {threshold}.")
        trainer.generate_submission()
    else:
        print(
            f"\nValidation metric {val_mcrmse} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
