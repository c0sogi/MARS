import os
import sys
import torch
import pandas as pd
import numpy as np
import importlib
import library.config
import library.utils
import library.data
import library.model
import library.train

# Cite debug_lesson_7
importlib.reload(library.config)
importlib.reload(library.utils)
importlib.reload(library.data)
importlib.reload(library.model)
importlib.reload(library.train)

from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.train import Runner
from library.model import NSLHN
from library.data import get_dataloaders


def main():
    # ==========================================
    # 1. Configuration Adjustments for Fast Baseline
    # ==========================================
    # Reduce epochs to ensure execution within 2 hours while allowing convergence
    Config.EPOCHS = 30
    Config.PATIENCE = 6

    print("Initializing Fast Baseline Run...")
    seed_everything(Config.SEED)

    # ==========================================
    # 2. Training Pipeline
    # ==========================================
    # Runner handles data loading, model init, and training loop
    runner = Runner(debug=False)
    runner.train()

    # ==========================================
    # 3. Validation & Failure Analysis
    # ==========================================
    print("\n" + "=" * 40)
    print("VALIDATION & FAILURE ANALYSIS")
    print("=" * 40)

    # Load the best model
    device = Config.DEVICE
    model = NSLHN().to(device)
    best_model_path = runner.best_model_path

    if not os.path.exists(best_model_path):
        print("Error: Best model checkpoint not found.")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Use the validation loader from the runner
    val_loader = runner.val_loader

    all_true = []
    all_pred = []
    all_sigma = []

    # Inference on Validation Set
    with torch.no_grad():
        for batch in val_loader:
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            week = batch["week"].to(device)
            base_week = batch["base_week"].to(device)
            y_true = batch["fvc"].to(device)

            # Forward pass
            y_pred, sigma = model(axial, coronal, tabular, base_fvc, week, base_week)

            all_true.extend(y_true.cpu().numpy())
            all_pred.extend(y_pred.cpu().numpy())
            all_sigma.extend(sigma.cpu().numpy())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_sigma = np.array(all_sigma)

    # Compute and Print Metric
    metric = laplace_log_likelihood_metric(all_true, all_pred, all_sigma)
    print(f"Final Validation Metric: {metric}")

    # Failure Analysis: Correlation of Error with Features
    # Retrieve the validation dataframe to access original features
    val_df = val_loader.dataset.df.copy()

    # Calculate absolute error
    val_df["pred_fvc"] = all_pred
    val_df["abs_error"] = np.abs(val_df["FVC"] - val_df["pred_fvc"])

    # Features to analyze
    analysis_features = [
        "Weeks",
        "Percent",
        "Age",
        "Baseline_FVC",
        "Baseline_Percent",
        "Baseline_Age",
    ]

    print("\nCorrelation between Input Features and Absolute Error:")
    # Compute correlation
    correlations = (
        val_df[analysis_features + ["abs_error"]]
        .corr()["abs_error"]
        .sort_values(ascending=False)
    )
    print(correlations.drop("abs_error"))

    # ==========================================
    # 4. Submission Generation
    # ==========================================
    THRESHOLD = -6.510164260864258

    if metric > THRESHOLD:
        print("\n" + "=" * 40)
        print("GENERATING SUBMISSION")
        print("=" * 40)

        # We need the test_loader. get_dataloaders returns (train, val, test)
        # We call it again to get the test loader properly initialized
        print("Loading test data...")
        _, _, test_loader = get_dataloaders(debug=False)

        submission_rows = []

        with torch.no_grad():
            for batch in test_loader:
                axial = batch["axial"].to(device)
                coronal = batch["coronal"].to(device)
                tabular = batch["tabular"].to(device)
                base_fvc = batch["base_fvc"].to(device)
                week = batch["week"].to(device)
                base_week = batch["base_week"].to(device)
                patient_weeks = batch["patient_week"]

                # Inference
                y_pred, sigma = model(
                    axial, coronal, tabular, base_fvc, week, base_week
                )

                y_pred = y_pred.cpu().numpy()
                sigma = sigma.cpu().numpy()

                # Collect results
                for pw, fvc, conf in zip(patient_weeks, y_pred, sigma):
                    submission_rows.append(
                        {"Patient_Week": pw, "FVC": fvc, "Confidence": conf}
                    )

        # Create DataFrame and Save
        sub_df = pd.DataFrame(submission_rows)

        # Ensure correct column order
        sub_df = sub_df[["Patient_Week", "FVC", "Confidence"]]

        save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        sub_df.to_csv(save_path, index=False)
        print(f"Submission saved successfully to: {save_path}")
        print(f"Total predictions: {len(sub_df)}")

    else:
        print(
            f"\nMetric {metric} did not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
