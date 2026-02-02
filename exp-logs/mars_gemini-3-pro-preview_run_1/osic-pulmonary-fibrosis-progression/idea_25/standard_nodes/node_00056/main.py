import sys
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Ensure local library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_loss
from library.dataset import LungDataset
from library.model import IASDANet
from library.engine import train_model, evaluate, predict


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between model error and input features.
    """
    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)

    model.eval()

    data_records = []

    with torch.no_grad():
        for batch in val_loader:
            # Move inputs to device
            axial_img = batch["axial_img"].to(device)
            coronal_img = batch["coronal_img"].to(device)
            tabular = batch["tabular"].to(device)
            time_delta = batch["time_delta"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            target = batch["target"].to(device)

            # Forward pass
            fvc_pred, sigma_pred = model(
                axial_img=axial_img,
                coronal_img=coronal_img,
                tabular=tabular,
                time_delta=time_delta,
                baseline_fvc=baseline_fvc,
            )

            # Move to CPU for analysis
            fvc_pred_np = fvc_pred.cpu().numpy().flatten()
            target_np = target.cpu().numpy().flatten()
            tabular_np = tabular.cpu().numpy()  # [B, 6]

            # Calculate Absolute Error
            abs_error = np.abs(target_np - fvc_pred_np)

            # Extract features (Indices based on Dataset.__getitem__)
            # 0: Age (norm), 1: Sex, 2-4: Smoke OHE, 5: Percent (norm)
            for i in range(len(abs_error)):
                record = {
                    "Error": abs_error[i],
                    "Age_Norm": tabular_np[i, 0],
                    "Sex": tabular_np[i, 1],
                    "Percent_Norm": tabular_np[i, 5],
                    # Reconstruct smoking status (0: Ex, 1: Never, 2: Current)
                    "Smoking_Ex": tabular_np[i, 2],
                    "Smoking_Never": tabular_np[i, 3],
                    "Smoking_Current": tabular_np[i, 4],
                }
                data_records.append(record)

    # Create DataFrame
    df_analysis = pd.DataFrame(data_records)

    # Calculate Correlations
    correlations = df_analysis.corr()["Error"].sort_values(ascending=False)

    print("Correlation between Absolute Error and Input Features:")
    print(correlations)
    print("-" * 40)

    # Identify worst performing group
    print("Top 5 Highest Errors:")
    print(df_analysis.sort_values("Error", ascending=False).head(5))


def main():
    # 1. Initialization
    Config.initialize()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Running Experiment: {Config.EXPERIMENT_NAME}")
    print(f"Device: {device}")

    # 2. Data Loading
    print("Initializing DataLoaders...")

    # Train Dataset
    train_dataset = LungDataset(mode="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Validation Dataset
    val_dataset = LungDataset(mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Training
    # The engine handles model instantiation, training loop, and saving best model
    train_model(train_loader, val_loader)

    # 4. Final Evaluation
    print("\nLoading best model for final evaluation...")
    model = IASDANet()
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    final_metric = evaluate(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 6. Submission
    SUBMISSION_THRESHOLD = -6.510164260864258

    if final_metric > SUBMISSION_THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )

        test_dataset = LungDataset(mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        predict(test_loader)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({SUBMISSION_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
