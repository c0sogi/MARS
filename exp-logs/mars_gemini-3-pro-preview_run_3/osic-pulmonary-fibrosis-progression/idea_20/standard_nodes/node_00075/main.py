import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from provided library
from library.config import Config
from library.model import TSCRNet
from library.data import get_dataloaders
from library.train import run_training
from library.utils import (
    seed_everything,
    inverse_scale_predictions,
    laplace_log_likelihood_metric,
)


def main():
    # 1. Setup and Configuration Override for Fast Baseline
    seed_everything(Config.SEED)

    # Override Config for faster execution within time limits
    # 20 epochs is sufficient for the model to converge on this dataset size
    # while keeping the runtime short.
    Config.N_EPOCHS = 20

    print(f"Configuration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Epochs: {Config.N_EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")

    # 2. Train the Model
    print("\n=== Starting Training Phase ===")
    # run_training handles the training loop and saves the best model to Config.CHECKPOINT_DIR
    run_training(debug=False)

    # 3. Load Best Model for Inference
    print("\n=== Loading Best Model ===")
    model = TSCRNet()
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()

    # 4. Validation Inference & Metric Calculation
    print("\n=== Running Validation Inference ===")
    _, val_loader, _ = get_dataloaders(debug=False)

    val_patient_weeks = []
    val_preds_mu = []
    val_preds_sigma = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(Config.DEVICE)
            tabular = batch["tabular"].to(Config.DEVICE)
            time_abs = batch["time_abs"].to(Config.DEVICE)
            targets = batch["target"].to(Config.DEVICE)
            p_weeks = batch["patient_week"]

            # Forward pass
            mu, sigma = model(images, tabular, time_abs)

            val_preds_mu.append(mu.cpu())
            val_preds_sigma.append(sigma.cpu())
            val_targets.append(targets.cpu())
            val_patient_weeks.extend(p_weeks)

    # Concatenate results
    val_preds_mu = torch.cat(val_preds_mu)
    val_preds_sigma = torch.cat(val_preds_sigma)
    val_targets = torch.cat(val_targets)

    # Inverse Scale to original units (ml)
    val_mu_orig, val_sigma_orig = inverse_scale_predictions(
        val_preds_mu, val_preds_sigma
    )
    val_targets_orig = val_targets * Config.TARGET_STD + Config.TARGET_MEAN

    # Calculate Metric
    final_metric = laplace_log_likelihood_metric(
        val_targets_orig, val_mu_orig, val_sigma_orig
    )
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n=== Performing Failure Analysis ===")
    # Create DataFrame for analysis
    val_mu_np = val_mu_orig.numpy().flatten()
    val_sigma_np = val_sigma_orig.numpy().flatten()
    val_targets_np = val_targets_orig.numpy().flatten()

    analysis_df = pd.DataFrame(
        {
            "Patient_Week": val_patient_weeks,
            "FVC_True": val_targets_np,
            "FVC_Pred": val_mu_np,
            "Sigma_Pred": val_sigma_np,
            "Abs_Error": np.abs(val_targets_np - val_mu_np),
        }
    )

    # Parse Patient and Weeks from Patient_Week string (Format: ID..._Weeks)
    split_data = analysis_df["Patient_Week"].str.rsplit("_", n=1, expand=True)
    analysis_df["Patient"] = split_data[0]
    analysis_df["Weeks"] = split_data[1].astype(int)

    # Load Validation Metadata to get input features
    val_meta = pd.read_csv(Config.VAL_META_PATH)

    # Merge features onto analysis dataframe
    full_analysis_df = pd.merge(
        analysis_df, val_meta, on=["Patient", "Weeks"], how="left"
    )

    # Calculate correlations
    features_to_check = ["Age", "Weeks", "Percent"]

    print("Correlation between Absolute Error and Features:")
    for feat in features_to_check:
        if feat in full_analysis_df.columns:
            # Drop NaNs if any to ensure pearsonr works
            valid_df = full_analysis_df[[feat, "Abs_Error"]].dropna()
            if len(valid_df) > 1:
                corr, _ = pearsonr(valid_df[feat], valid_df["Abs_Error"])
                print(f"  {feat}: {corr:.4f}")

    # 6. Submission Generation
    threshold = -6.573619738753321
    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric:.6f}) > Threshold ({threshold:.6f}). Generating submission..."
        )

        _, _, test_loader = get_dataloaders(debug=False)

        test_patient_weeks = []
        test_preds_mu = []
        test_preds_sigma = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(Config.DEVICE)
                tabular = batch["tabular"].to(Config.DEVICE)
                time_abs = batch["time_abs"].to(Config.DEVICE)
                p_weeks = batch["patient_week"]

                mu, sigma = model(images, tabular, time_abs)

                test_preds_mu.append(mu.cpu())
                test_preds_sigma.append(sigma.cpu())
                test_patient_weeks.extend(p_weeks)

        # Concatenate
        test_preds_mu = torch.cat(test_preds_mu)
        test_preds_sigma = torch.cat(test_preds_sigma)

        # Inverse Scale
        test_mu_orig, test_sigma_orig = inverse_scale_predictions(
            test_preds_mu, test_preds_sigma
        )

        # Convert to numpy
        test_mu_np = test_mu_orig.numpy().flatten()
        test_sigma_np = test_sigma_orig.numpy().flatten()

        # Apply Clipping for Submission (Confidence >= 70ml)
        test_sigma_clipped = np.maximum(test_sigma_np, 70)

        # Create Submission DataFrame
        sub_df = pd.DataFrame(
            {
                "Patient_Week": test_patient_weeks,
                "FVC": test_mu_np,
                "Confidence": test_sigma_clipped,
            }
        )

        # Ensure output directory exists and save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        sub_path = Config.SUBMISSION_PATH
        sub_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
        print(sub_df.head())

    else:
        print(
            f"\nMetric ({final_metric:.6f}) <= Threshold ({threshold:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
