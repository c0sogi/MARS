import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.dataset import LungDataset
from library.model import ChannelAdaptiveDualAxisNet
from library.loss import LaplaceLogLikelihoodLoss
from library.train_eval import train_model, set_seed


def main():
    # 1. Setup
    print("Setting up configuration and environment...")
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Load Metadata
    print("Loading metadata...")
    try:
        train_df = pd.read_csv(Config.TRAIN_CSV)
        val_df = pd.read_csv(Config.VAL_CSV)
    except FileNotFoundError as e:
        print(f"Error loading metadata: {e}")
        sys.exit(1)

    # 3. Train Model
    # Using the provided training function which handles the loop and saving best model
    # We use the full dataset (debug=False) because it is small (~1k samples)
    # and training is fast enough (minutes) to fit within the time limit.
    print("Starting training pipeline...")
    best_model_path = train_model(train_df, val_df, debug=False)

    # 4. Validation Inference & Metric Calculation
    print("\nRunning final validation inference...")

    # Load best model
    model = ChannelAdaptiveDualAxisNet()
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    # Setup Validation Loader
    val_dataset = LungDataset(val_df, mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_metrics = []
    val_errors = []
    val_features = []

    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)

            target_fvc = batch["fvc"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            week_delta = batch["week"].to(device)

            # Forward pass
            alpha, sigma_base, sigma_growth = model(axial, coronal, tabular)

            # Reconstruct Predictions
            pred_fvc = base_fvc + alpha * week_delta
            pred_sigma = sigma_base + sigma_growth * torch.abs(week_delta)

            # Calculate Metric components
            # Metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
            delta = torch.abs(target_fvc - pred_fvc)
            delta_clipped = torch.clamp(delta, max=1000.0)
            sigma_clipped = torch.clamp(pred_sigma, min=70.0)

            sqrt_2 = np.sqrt(2)
            metric_batch = -(sqrt_2 * delta_clipped) / sigma_clipped - torch.log(
                sqrt_2 * sigma_clipped
            )

            # Store results
            val_metrics.extend(metric_batch.cpu().numpy())
            val_errors.extend(delta.cpu().numpy())
            val_features.append(tabular.cpu().numpy())

    # Compute Final Metric
    final_metric = np.mean(val_metrics)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    val_features = np.concatenate(val_features, axis=0)
    # Feature order in dataset.py: Age, Sex, Smk_Ex, Smk_Never, Smk_Current, Percent
    feature_names = [
        "Age_Norm",
        "Sex_Female",
        "Smk_Ex",
        "Smk_Never",
        "Smk_Current",
        "Percent_Norm",
    ]

    analysis_df = pd.DataFrame(val_features, columns=feature_names)
    analysis_df["Error_Magnitude"] = val_errors

    # Calculate correlations
    correlations = analysis_df.corr()["Error_Magnitude"].drop("Error_Magnitude")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # 6. Submission Generation
    threshold = -6.510164260864258

    if final_metric > threshold:
        print(f"\nMetric {final_metric} > {threshold}. Generating submission...")

        try:
            test_df = pd.read_csv(Config.TEST_CSV)
        except FileNotFoundError:
            print("Test metadata not found.")
            sys.exit(1)

        test_dataset = LungDataset(test_df, mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        all_preds_fvc = []
        all_preds_sigma = []

        with torch.no_grad():
            for batch in test_loader:
                axial = batch["axial"].to(device)
                coronal = batch["coronal"].to(device)
                tabular = batch["tabular"].to(device)
                base_fvc = batch["base_fvc"].to(device)
                week_delta = batch["week"].to(device)

                alpha, sigma_base, sigma_growth = model(axial, coronal, tabular)

                pred_fvc = base_fvc + alpha * week_delta
                pred_sigma = sigma_base + sigma_growth * torch.abs(week_delta)

                all_preds_fvc.extend(pred_fvc.cpu().numpy())
                all_preds_sigma.extend(pred_sigma.cpu().numpy())

        # Create submission DataFrame
        # Note: test_df from metadata/test.csv is aligned with sample_submission rows
        sub_df = pd.DataFrame(
            {
                "Patient_Week": test_df["Patient_Week"],
                "FVC": all_preds_fvc,
                "Confidence": all_preds_sigma,
            }
        )

        # Save
        sub_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

    else:
        print(
            f"\nMetric {final_metric} <= {threshold}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
