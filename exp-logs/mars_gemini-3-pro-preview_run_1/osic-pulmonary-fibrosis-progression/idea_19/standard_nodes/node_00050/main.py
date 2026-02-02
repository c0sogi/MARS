import sys
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood
from library.dataset import LungDataset
from library.model import TabularGatedDualViewNetwork
from library.engine import fit


def run_validation_and_analysis(model, val_loader, val_df, device):
    """
    Runs inference on validation set, computes metric, and performs failure analysis.
    """
    model.eval()

    all_targets = []
    all_preds = []
    all_sigmas = []

    # Inference loop
    with torch.no_grad():
        for data in val_loader:
            image_axial = data["image_axial"].to(device)
            image_coronal = data["image_coronal"].to(device)
            tabular = data["tabular"].to(device)
            dt = data["dt"].to(device)
            baseline_fvc = data["baseline_fvc"].to(device)
            target = data["target"].to(device)

            outputs = model(image_axial, image_coronal, tabular, dt, baseline_fvc)

            all_targets.append(target.cpu().numpy())
            all_preds.append(outputs["fvc_pred"].cpu().numpy())
            all_sigmas.append(outputs["confidence_pred"].cpu().numpy())

    # Concatenate results
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    sigma = np.concatenate(all_sigmas)

    # 1. Compute Metric
    metric_score = laplace_log_likelihood(y_true, y_pred, sigma)

    # 2. Failure Analysis
    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)

    # Calculate absolute error
    abs_error = np.abs(y_true - y_pred)

    # Create analysis dataframe using the original validation metadata
    # We rely on the fact that DataLoader with shuffle=False preserves order
    analysis_df = val_df.copy()
    analysis_df["abs_error"] = abs_error
    analysis_df["pred_fvc"] = y_pred
    analysis_df["pred_sigma"] = sigma

    # Select features for correlation
    features_to_analyze = [
        "Baseline_Age",
        "Baseline_Percent",
        "Baseline_FVC",
        "Weeks",
        "dt",
    ]

    # Compute correlations
    print("Correlation between Absolute Error and Input Features:")
    correlations = analysis_df[features_to_analyze].corrwith(analysis_df["abs_error"])
    print(correlations.sort_values(ascending=False))

    return metric_score


def generate_submission(model, device):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    print("\nGenerating submission...")

    # Load Test Data
    test_ds = LungDataset(mode="test", load_cached_data=True)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    model.eval()
    all_preds = []
    all_sigmas = []

    with torch.no_grad():
        for data in test_loader:
            image_axial = data["image_axial"].to(device)
            image_coronal = data["image_coronal"].to(device)
            tabular = data["tabular"].to(device)
            dt = data["dt"].to(device)
            baseline_fvc = data["baseline_fvc"].to(device)

            outputs = model(image_axial, image_coronal, tabular, dt, baseline_fvc)

            all_preds.append(outputs["fvc_pred"].cpu().numpy())
            all_sigmas.append(outputs["confidence_pred"].cpu().numpy())

    # Concatenate
    y_pred = np.concatenate(all_preds)
    sigma = np.concatenate(all_sigmas)

    # Prepare Submission DataFrame
    # test_ds.df corresponds to metadata/test.csv which has Patient_Week
    sub_df = test_ds.df.copy()
    sub_df["FVC"] = y_pred

    # Clip confidence at 70 as per metric/task requirements
    sub_df["Confidence"] = np.maximum(sigma, 70.0)

    # Format output
    submission = sub_df[["Patient_Week", "FVC", "Confidence"]]

    # Save
    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
    print(submission.head())


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    train_ds = LungDataset(mode="train", load_cached_data=True)
    val_ds = LungDataset(mode="val", load_cached_data=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing Tabular-Gated Dual-View Network...")
    model = TabularGatedDualViewNetwork().to(device)

    # 4. Training
    # Limiting epochs to 30 for a fast baseline execution
    print("Starting training...")
    fit(model, train_loader, val_loader, device, epochs=30, patience=Config.PATIENCE)

    # 5. Evaluation & Analysis
    print("Loading best model for analysis...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    val_score = run_validation_and_analysis(model, val_loader, val_ds.df, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_score}")

    # 6. Submission
    THRESHOLD = -6.510164260864258

    if val_score > THRESHOLD:
        generate_submission(model, device)
    else:
        print(
            f"Validation score {val_score} is not higher than threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
