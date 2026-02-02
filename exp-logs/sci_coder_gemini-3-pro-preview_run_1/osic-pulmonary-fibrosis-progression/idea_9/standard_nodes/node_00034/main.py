import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Import from provided library files
from library.config import Config
from library.train import run_training
from library.model import DualAxisTransformer
from library.data import get_dataloaders
from library.utils import seed_everything, LaplaceLogLikelihood

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for a fast baseline execution
    # The dataset is small (1109 train samples), so 20 epochs is very fast (<10 mins)
    Config.EPOCHS = 20
    Config.PATIENCE = 5
    Config.BATCH_SIZE = 16

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Create submission directory
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    print("Configuration configured for fast baseline.")
    print(f"Epochs: {Config.EPOCHS}, Batch Size: {Config.BATCH_SIZE}")

    # ==========================================
    # 2. Training
    # ==========================================
    print("\nStarting Training Pipeline...")
    # run_training handles data loading, model init, training loop, and saving best_model.pth
    run_training()

    # ==========================================
    # 3. Validation & Evaluation
    # ==========================================
    print("\nStarting Validation & Failure Analysis...")

    device = torch.device(Config.DEVICE)

    # Load Validation Data
    # We use debug=False to ensure we validate on the full validation set for accurate metrics
    _, val_loader, _ = get_dataloaders(debug=False)

    # Load Best Model
    model = DualAxisTransformer()
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Containers for analysis
    all_fvc_true = []
    all_fvc_pred = []
    all_sigma_pred = []
    all_errors = []

    # Inference Loop (No Gradients)
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)
            meta = batch["meta"].to(device)
            target = batch["target"].to(device)

            # Forward pass
            preds = model(img_ax, img_cor, tabular)

            alpha = preds[:, 0]
            sigma_base = preds[:, 1]
            sigma_growth = preds[:, 2]

            delta_week = meta[:, 0]
            base_fvc = meta[:, 1]

            # Calculate predictions
            fvc_pred = base_fvc + alpha * delta_week
            sigma_pred = sigma_base + sigma_growth * torch.abs(delta_week)

            # Store results
            all_fvc_true.append(target.cpu())
            all_fvc_pred.append(fvc_pred.cpu())
            all_sigma_pred.append(sigma_pred.cpu())

    # Concatenate results
    y_true = torch.cat(all_fvc_true)
    y_pred = torch.cat(all_fvc_pred)
    sigma = torch.cat(all_sigma_pred)

    # Compute Final Metric
    # Note: LaplaceLogLikelihood returns a scalar tensor
    final_metric = LaplaceLogLikelihood(y_true, y_pred, sigma).item()

    # Print the required metric string
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\nPerforming Failure Analysis...")

    # Calculate absolute errors
    errors = torch.abs(y_true - y_pred).numpy()

    # Get Validation DataFrame to access metadata
    # Since shuffle=False for val_loader, the order matches the dataset dataframe
    val_df = val_loader.dataset.df.copy()

    # Ensure lengths match
    if len(val_df) != len(errors):
        print("Warning: Mismatch in validation set size and predictions.")
        val_df = val_df.iloc[: len(errors)]

    val_df["Error"] = errors
    val_df["Predicted_FVC"] = y_pred.numpy()
    val_df["Sigma"] = sigma.numpy()

    # Select features for correlation analysis
    # We want to see if Error correlates with Age, Percent, Weeks, etc.
    analysis_cols = ["Age", "Percent", "Weeks", "FVC"]

    # Compute correlations
    correlations = val_df[analysis_cols].corrwith(val_df["Error"])

    print("Correlation between Absolute Error and Features:")
    print(correlations)

    # Identify systematic bias
    mean_error = np.mean(errors)
    print(f"Mean Absolute Error: {mean_error:.4f} ml")

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating Submission..."
        )

        # Load Test Data
        _, _, test_loader = get_dataloaders(debug=False)

        results = []

        with torch.no_grad():
            for batch in test_loader:
                img_ax = batch["img_ax"].to(device)
                img_cor = batch["img_cor"].to(device)
                tabular = batch["tabular"].to(device)
                meta = batch["meta"].to(device)
                patient_weeks = batch["patient_week"]

                preds = model(img_ax, img_cor, tabular)

                alpha = preds[:, 0]
                sigma_base = preds[:, 1]
                sigma_growth = preds[:, 2]

                delta_week = meta[:, 0]
                base_fvc = meta[:, 1]

                fvc_pred = base_fvc + alpha * delta_week
                sigma_pred = sigma_base + sigma_growth * torch.abs(delta_week)

                fvc_pred = fvc_pred.cpu().numpy()
                sigma_pred = sigma_pred.cpu().numpy()

                for i, pw in enumerate(patient_weeks):
                    # Clip confidence as per submission requirement (though metric handles it, good practice)
                    conf = max(sigma_pred[i], 70.0)
                    results.append(
                        {"Patient_Week": pw, "FVC": fvc_pred[i], "Confidence": conf}
                    )

        # Save Submission
        sub_df = pd.DataFrame(results)
        sub_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
        print(sub_df.head())

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
