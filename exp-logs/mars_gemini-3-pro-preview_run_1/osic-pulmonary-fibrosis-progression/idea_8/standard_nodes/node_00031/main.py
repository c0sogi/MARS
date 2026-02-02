import os
import sys
import torch
import pandas as pd
import numpy as np
from unittest.mock import MagicMock


# 1. Patch tqdm to suppress progress bars
import tqdm

# Capture original class to allow inheritance
OriginalTqdm = tqdm.tqdm


class SilentTqdm(OriginalTqdm):
    def __init__(self, *args, **kwargs):
        # Force disable=True to suppress output
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


# Patch both the standard tqdm and tqdm.auto which is used in library files
tqdm.tqdm = SilentTqdm
import tqdm.auto

tqdm.auto.tqdm = SilentTqdm

# 2. Import Library Modules
from library.config import Config
from library.train import run_training
from library.predict import generate_predictions
from library.data import get_dataloaders
from library.model import AttentionFusedDualAxisNet
from library.utils import seed_everything, laplace_log_likelihood


def main():
    # Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Configure for fast baseline
    # 20 epochs is sufficient for convergence on this small dataset
    # while keeping runtime well within the limit.
    Config.EPOCHS = 20

    # 3. Training
    print("Starting training...")
    # run_training handles training, SWA, checkpointing, and saves 'best_model.pth'
    run_training(epochs=Config.EPOCHS)

    # 4. Validation & Metric Calculation
    print("Performing validation inference...")

    # Load Validation Data
    # We ignore train/test loaders here
    _, val_loader, _ = get_dataloaders(Config)

    # Load Best Model
    model = AttentionFusedDualAxisNet().to(device)
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(best_model_path):
        print("Critical Error: Best model weights not found.")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Inference Loop
    all_fvc_true = []
    all_fvc_pred = []
    all_sigma_pred = []

    with torch.no_grad():
        for batch in val_loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)
            relative_week = batch["relative_week"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            fvc_target = batch["fvc_target"].to(device)

            fvc_pred, sigma_pred = model(
                tabular, img_ax, img_cor, relative_week, baseline_fvc
            )

            all_fvc_true.append(fvc_target.cpu().numpy())
            all_fvc_pred.append(fvc_pred.cpu().numpy())
            all_sigma_pred.append(sigma_pred.cpu().numpy())

    y_true = np.concatenate(all_fvc_true)
    y_pred = np.concatenate(all_fvc_pred)
    sigma = np.concatenate(all_sigma_pred)

    # Compute Final Metric
    final_metric = laplace_log_likelihood(y_true, y_pred, sigma)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("Performing failure analysis...")

    # Reconstruct validation dataframe to align features with predictions
    # Logic matches library.data.get_dataloaders preprocessing
    val_df = pd.read_csv(Config.VAL_CSV)
    val_df = val_df.sort_values(["Patient", "Weeks"])

    if len(val_df) == len(y_pred):
        val_df["Pred_FVC"] = y_pred
        val_df["Abs_Error"] = np.abs(val_df["FVC"] - val_df["Pred_FVC"])

        # Calculate correlations
        features = ["Age", "Percent", "Weeks"]
        correlations = (
            val_df[features + ["Abs_Error"]].corr()["Abs_Error"].drop("Abs_Error")
        )

        print("Correlation between Abs_Error and features:")
        print(correlations)
    else:
        print(
            f"Warning: Validation DataFrame length ({len(val_df)}) mismatch with predictions ({len(y_pred)}). Skipping correlation analysis."
        )

    # 6. Submission Logic
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"Metric ({final_metric}) > Threshold ({THRESHOLD}). Generating final submission..."
        )
        generate_predictions(weights_path=best_model_path)
    else:
        print(
            f"Metric ({final_metric}) <= Threshold ({THRESHOLD}). Submission discarded."
        )
        # run_training might have created a submission file; delete it to strictly enforce "If and only if"
        if os.path.exists(Config.SUBMISSION_PATH):
            os.remove(Config.SUBMISSION_PATH)


if __name__ == "__main__":
    main()
