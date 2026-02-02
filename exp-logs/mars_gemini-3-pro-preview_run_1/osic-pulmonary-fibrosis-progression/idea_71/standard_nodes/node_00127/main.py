import os
import sys
import torch
import numpy as np
import pandas as pd

from library.config import Config, seed_everything
from library.train import run_training
from library.model import DBSLNet
from library.data import get_dataloaders
from library.utils import metric_laplace_log_likelihood
from library.predict import run_prediction


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Training
    # We limit epochs to 25 to ensure fast execution within the time limit.
    # The dataset is small, so convergence usually happens quickly.
    print("Starting Training Pipeline...")
    run_training(epochs=25)

    # 3. Validation Inference
    print("Starting Validation Inference...")

    # Initialize model and load best weights
    model = DBSLNet()
    model.to(device)

    if os.path.exists(Config.MODEL_SAVE_PATH):
        checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print("CRITICAL: Best model checkpoint not found. Validation cannot proceed.")
        return

    model.eval()

    # Get validation dataloader
    # Note: val_loader has shuffle=False
    _, val_loader = get_dataloaders()

    all_targets = []
    all_fvc_preds = []
    all_sigma_preds = []
    all_meta_features = []

    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            target = batch["target"].to(device)
            week = batch["week"].to(device)
            base_week = batch["base_week"].to(device)
            base_fvc = batch["base_fvc"].to(device)

            # Forward pass
            fvc_pred, sigma_pred = model(
                img_ax, img_cor, tabular, week, base_week, base_fvc
            )

            # Collect results
            all_targets.append(target.cpu().numpy())
            all_fvc_preds.append(fvc_pred.cpu().numpy())
            all_sigma_preds.append(sigma_pred.cpu().numpy())
            all_meta_features.append(tabular.cpu().numpy())

    # Concatenate results
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_fvc_preds)
    sigma = np.concatenate(all_sigma_preds)
    meta_matrix = np.concatenate(all_meta_features)

    # 4. Metric Calculation
    # Using the provided utility function
    final_metric = metric_laplace_log_likelihood(y_true, y_pred, sigma)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate Error Magnitude
    abs_error = np.abs(y_true - y_pred)

    # Create DataFrame for correlation analysis
    # Tabular features in tensor are: [Age, Percent, Sex, Smoke] (Normalized/Encoded)
    # Correlation is invariant to linear scaling, so normalized values are fine.
    analysis_df = pd.DataFrame(
        {
            "Error_Magnitude": abs_error,
            "Age": meta_matrix[:, 0],
            "Percent": meta_matrix[:, 1],
            "Sex": meta_matrix[:, 2],
            "SmokingStatus": meta_matrix[:, 3],
            "Predicted_Confidence": sigma,
        }
    )

    # Compute correlations
    correlations = analysis_df.corr()["Error_Magnitude"].drop("Error_Magnitude")

    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # 6. Submission Generation
    # Threshold defined in task
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\nValidation Metric ({final_metric}) is better than threshold ({THRESHOLD})."
        )
        print("Generating submission for test set...")
        run_prediction()
    else:
        print(
            f"\nValidation Metric ({final_metric}) did not meet threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
