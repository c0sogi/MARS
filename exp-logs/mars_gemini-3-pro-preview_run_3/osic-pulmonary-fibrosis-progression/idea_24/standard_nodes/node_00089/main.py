import sys
import os
import numpy as np
import torch
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.utils import seed_everything, metric_laplace_log_likelihood
from library.data import get_dataloaders
from library.model import MACLINet
from library.train import Trainer


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # Modify Config for fast baseline execution
    # Reducing epochs to ensure completion within time limits while maintaining performance
    Config.EPOCHS = 25

    # 2. Load Data
    print("Loading and preprocessing data...")
    # load_cached_data=True ensures we use pre-generated .npy files if available
    train_loader, val_loader, test_loader, scalers = get_dataloaders(
        load_cached_data=True
    )

    # 3. Initialize Model
    print("Initializing MACLI-Net...")
    model = MACLINet()

    # 4. Train
    print("Starting training...")
    trainer = Trainer(model, train_loader, val_loader, test_loader, scalers)
    trainer.fit()

    # 5. Validation & Failure Analysis
    print("Performing validation inference...")

    # We need to manually run inference on val_loader to get arrays for analysis
    # Trainer.validate() only returns the score, but we need predictions for failure analysis
    model.eval()
    device = Config.DEVICE

    all_mu = []
    all_sigma = []
    all_targets = []
    all_clinical = []

    with torch.no_grad():
        for (images, clinical), targets in val_loader:
            images = images.to(device)
            # clinical is needed for model input
            clinical_gpu = clinical.to(device)

            mu, sigma = model(images, clinical_gpu)

            all_mu.append(mu.cpu().numpy())
            all_sigma.append(sigma.cpu().numpy())
            all_targets.append(targets.numpy())
            all_clinical.append(clinical.numpy())

    all_mu = np.concatenate(all_mu)
    all_sigma = np.concatenate(all_sigma)
    all_targets = np.concatenate(all_targets)
    all_clinical = np.concatenate(all_clinical, axis=0)

    # Inverse Transform
    if scalers and "target" in scalers:
        scaler = scalers["target"]
        mu_real = scaler.inverse_transform(all_mu.reshape(-1, 1)).flatten()
        target_real = scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()
        # Scale sigma (standard deviation scales linearly)
        sigma_real = all_sigma.flatten() * scaler.scale_[0]
    else:
        mu_real = all_mu.flatten()
        target_real = all_targets.flatten()
        sigma_real = all_sigma.flatten()

    # Calculate Metric
    val_score = metric_laplace_log_likelihood(target_real, mu_real, sigma_real)
    print(f"Final Validation Metric: {val_score}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    errors = np.abs(target_real - mu_real)

    # Feature names corresponding to data.py logic:
    # [Baseline_FVC, Baseline_Percent, Age, Sex_Code, Smoking_Code, Relative_Time]
    feature_names = [
        "Baseline_FVC",
        "Baseline_Percent",
        "Age",
        "Sex_Code",
        "Smoking_Code",
        "Relative_Time",
    ]

    print("Correlation between Absolute Error and Features:")
    for i, name in enumerate(feature_names):
        feat_vals = all_clinical[:, i]
        # Check for zero variance to avoid division by zero in correlation
        if np.std(feat_vals) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(errors, feat_vals)[0, 1]
        print(f"  {name}: {corr:.4f}")

    # 6. Submission
    # Threshold from prompt requirements
    threshold = -6.573619738753321

    if val_score > threshold:
        print(
            f"\nValidation score ({val_score}) > threshold ({threshold}). Generating submission..."
        )
        # Trainer.predict() handles loading best model, inference on test_loader, inverse transform, and saving
        trainer.predict()
    else:
        print(
            f"\nValidation score ({val_score}) <= threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
