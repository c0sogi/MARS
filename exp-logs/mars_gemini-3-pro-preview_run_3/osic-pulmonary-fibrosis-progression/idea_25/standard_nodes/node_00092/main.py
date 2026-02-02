import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import from library
from library.config import Config
from library.utils import seed_everything, inverse_transform, laplace_log_likelihood
from library.data import get_train_val_datasets, get_test_dataset
from library.model import CCVRNet
from library.train import train_model

# -------------------------------------------------------------------------
# Configuration & Setup
# -------------------------------------------------------------------------
# Override Config for fast baseline execution
# 20 epochs is sufficient for a baseline check on this dataset size with A100
Config.EPOCHS = 20
Config.DEBUG = False


def run_inference(model, loader, device):
    """
    Runs inference on a dataloader.
    Returns:
        mu_ml: Predicted FVC in ml
        sigma_ml: Predicted Uncertainty in ml
        targets_ml: True FVC in ml (if available, else None)
    """
    model.eval()
    all_mu = []
    all_sigma = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            # Unpack batch depending on mode
            if len(batch) == 3:
                # Train/Val mode: image, tabular, target
                images, tabular, targets = batch
                targets = targets.to(device)
                all_targets.append(targets.cpu().numpy())
            else:
                # Test mode: image, tabular, patient_id
                images, tabular, _ = batch

            images = images.to(device)
            tabular = tabular.to(device)

            mu, sigma = model(images, tabular)

            all_mu.append(mu.cpu().numpy())
            all_sigma.append(sigma.cpu().numpy())

    # Concatenate
    mu_scaled = np.concatenate(all_mu)
    sigma_scaled = np.concatenate(all_sigma)

    # Inverse Transform
    mu_ml, sigma_ml = inverse_transform(mu_scaled, sigma_scaled)

    if all_targets:
        targets_scaled = np.concatenate(all_targets).flatten()
        # Inverse transform targets: target_ml = target_scaled * std + mean
        targets_ml = targets_scaled * Config.TARGET_STD + Config.TARGET_MEAN
        return mu_ml, sigma_ml, targets_ml
    else:
        return mu_ml, sigma_ml, None


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Train Model
    # This function handles the training loop and saves the best model to checkpoints/best_model.pth
    print("Starting training...")
    train_model(debug=Config.DEBUG)

    # 3. Load Best Model
    print("Loading best model for validation...")
    model = CCVRNet().to(device)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 4. Validation & Metric Calculation
    print("Performing validation...")
    _, val_ds = get_train_val_datasets()
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_mu, val_sigma, val_targets = run_inference(model, val_loader, device)

    # Compute Metric
    final_metric = laplace_log_likelihood(val_targets, val_mu, val_sigma)
    # Print full precision as required
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(val_targets - val_mu)

    # Get features from validation dataframe
    analysis_df = val_ds.df.copy()
    analysis_df["Error"] = errors

    # Features to check
    features = ["Age", "Weeks", "Baseline_FVC", "Percent", "Sex_Code", "Smoking_Code"]

    print("Correlation between Absolute Error and Features:")
    correlations = analysis_df[features + ["Error"]].corr()["Error"].drop("Error")
    print(correlations)

    # 6. Submission
    THRESHOLD = -6.573619738753321
    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Load test data
        test_ds, sub_df = get_test_dataset()
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Inference
        test_mu, test_sigma, _ = run_inference(model, test_loader, device)

        # Post-processing for submission
        # Clip confidence at 70ml as per metric definition
        test_sigma_clipped = np.maximum(test_sigma, 70)

        # Assign to dataframe
        sub_df["FVC"] = test_mu
        sub_df["Confidence"] = test_sigma_clipped

        # Save
        save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        sub_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
