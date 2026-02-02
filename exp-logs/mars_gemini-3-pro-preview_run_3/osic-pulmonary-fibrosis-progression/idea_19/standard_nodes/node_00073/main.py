import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

# Import from provided library files
from library.config import Config
from library.train import Trainer
from library.evaluate import predict_test_set
from library.utils import seed_everything, score_function
from library.model import MACRNet


def main():
    # 1. Setup and Configuration
    # Override Config for a fast baseline execution
    # Cite Lesson 00009: Extended training for convergence
    Config.EPOCHS = 30

    # Ensure reproducibility
    seed_everything(Config.SEED)

    print(f"Starting Fast Baseline Run (Epochs={Config.EPOCHS})...")

    # 2. Training
    trainer = Trainer()
    trainer.fit(epochs=Config.EPOCHS)

    # 3. Validation & Metric Calculation
    print("\nPerforming Final Validation and Failure Analysis...")

    # Load the best model
    device = torch.device(Config.DEVICE)
    model = MACRNet()
    model.to(device)

    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
        print("Loaded best model checkpoint.")
    else:
        print("Warning: Best model not found. Using current model state.")

    model.eval()

    # Storage for analysis
    all_targets = []
    all_mu = []
    all_sigma = []
    all_features = []  # To store tabular features

    # Retrieve stats for un-normalization
    stats = trainer.stats

    with torch.no_grad():
        for imgs, tabs, targets in trainer.val_loader:
            imgs = imgs.to(device)
            tabs = tabs.to(device)
            targets = targets.to(device).view(-1, 1)

            # Use AMP if enabled
            with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                preds = model(imgs, tabs)

            # Extract raw predictions
            mu_pred = preds[:, 0]
            raw_sigma = preds[:, 1]
            sigma_pred = F.softplus(raw_sigma) + 1e-6

            # Un-normalize for metric calculation
            # mu_orig = mu_pred * std + mean
            mu_orig = mu_pred * stats["FVC_std"] + stats["FVC_mean"]
            # sigma_orig = sigma_pred * std
            sigma_orig = sigma_pred * stats["FVC_std"]
            # targets_orig = target * std + mean
            targets_orig = targets.squeeze() * stats["FVC_std"] + stats["FVC_mean"]

            # Store for batch calculation
            all_targets.append(targets_orig.cpu().numpy())
            all_mu.append(mu_orig.cpu().numpy())
            all_sigma.append(sigma_orig.cpu().numpy())
            all_features.append(tabs.cpu().numpy())

    # Concatenate all batches
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_mu)
    sigma = np.concatenate(all_sigma)
    features = np.concatenate(all_features)

    # Calculate Final Metric
    final_metric = score_function(y_true, y_pred, sigma)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    # Calculate Absolute Error
    abs_error = np.abs(y_true - y_pred)

    # Features in 'tabs': [BaseFVC_norm, t_rel, Age_norm, Sex_Code, Smoking_Code]
    feature_names = ["Baseline_FVC", "Time_Rel", "Age", "Sex", "Smoking"]

    # Create DataFrame for correlation
    analysis_df = pd.DataFrame(features, columns=feature_names)
    analysis_df["Abs_Error"] = abs_error

    print("\nFailure Analysis (Correlation with Absolute Error):")
    correlations = analysis_df.corr()["Abs_Error"].drop("Abs_Error")
    print(correlations)

    # 5. Submission
    # Threshold from requirements
    THRESHOLD = -6.573619738753321

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        predict_test_set()
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
