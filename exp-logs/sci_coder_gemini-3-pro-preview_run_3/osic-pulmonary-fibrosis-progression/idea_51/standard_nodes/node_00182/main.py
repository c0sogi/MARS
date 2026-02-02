import os
import sys
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import (
    seed_everything,
    get_global_stats,
    laplace_log_likelihood_score,
)
from library.data import get_dataloaders
from library.model import ARLRNet
from library.train import Trainer


def main():
    # 1. Configuration & Setup
    # Override epochs for fast baseline execution within time limits
    Config.EPOCHS = 20
    Config.T_MAX = 20

    seed_everything(Config.SEED)
    Config.setup()

    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    print("Loading data...")
    # load_cached_data=True allows using preprocessed npy files if available
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Get global stats for normalization/denormalization
    global_mean, global_std = get_global_stats(Config.TRAIN_CSV)
    print(f"Global Stats: Mean={global_mean:.4f}, Std={global_std:.4f}")

    # 3. Model Initialization
    print("Initializing ARLRNet...")
    model = ARLRNet(global_std_target=global_std)
    model.to(device)

    # 4. Training
    print("Starting training...")
    trainer = Trainer(model, train_loader, val_loader, (global_mean, global_std))
    trainer.fit()

    # 5. Validation & Metric Calculation
    print("Performing final validation...")
    # Load best model weights
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model checkpoint.")
    else:
        print("Warning: Best model checkpoint not found. Using current model weights.")

    model.eval()
    val_preds_mu = []
    val_preds_sigma = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            stream_a = batch["stream_a"].to(device)
            stream_b = batch["stream_b"].to(device)
            targets_raw = batch["fvc_raw"].numpy()

            # Forward pass
            outputs = model(images, stream_a, stream_b)

            mu_scaled = outputs[:, 0].cpu().numpy()
            sigma_scaled = outputs[:, 1].cpu().numpy()

            # Inverse Transform
            mu_final = mu_scaled * global_std + global_mean
            sigma_final = sigma_scaled * global_std

            val_preds_mu.extend(mu_final)
            val_preds_sigma.extend(sigma_final)
            val_targets.extend(targets_raw)

    val_preds_mu = np.array(val_preds_mu)
    val_preds_sigma = np.array(val_preds_sigma)
    val_targets = np.array(val_targets)

    # Compute Metric
    final_metric = laplace_log_likelihood_score(
        val_targets, val_preds_mu, val_preds_sigma
    )
    print(f"Final Validation Metric: {final_metric:.16f}")

    # 6. Failure Analysis
    print("\nFailure Analysis:")
    # Calculate absolute error
    errors = np.abs(val_targets - val_preds_mu)

    # Get validation metadata
    val_df = val_loader.dataset.df.copy()

    if len(val_df) == len(errors):
        val_df["Error_Mag"] = errors

        # Features to check correlation against
        features = ["Weeks", "Base_FVC", "Age", "Percent"]
        correlations = {}

        print("Correlation between Error Magnitude and Features:")
        for feat in features:
            if feat in val_df.columns:
                corr = val_df[feat].corr(val_df["Error_Mag"])
                correlations[feat] = corr
                print(f"  {feat}: {corr:.4f}")
    else:
        print(
            "Warning: Validation dataset length mismatch. Skipping correlation analysis."
        )

    # 7. Submission Generation
    threshold = -6.573619738753321

    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric:.6f}) > Threshold ({threshold}). Generating submission..."
        )

        test_ids = []
        test_mu = []
        test_sigma = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                stream_a = batch["stream_a"].to(device)
                stream_b = batch["stream_b"].to(device)
                p_ids = batch["patient_week"]

                outputs = model(images, stream_a, stream_b)

                mu_scaled = outputs[:, 0].cpu().numpy()
                sigma_scaled = outputs[:, 1].cpu().numpy()

                # Inverse Transform
                mu_final = mu_scaled * global_std + global_mean
                sigma_final = sigma_scaled * global_std

                # Clip confidence at 70ml (Submission requirement)
                sigma_final = np.maximum(sigma_final, 70)

                test_ids.extend(p_ids)
                test_mu.extend(mu_final)
                test_sigma.extend(sigma_final)

        # Create submission DataFrame
        sub_df = pd.DataFrame(
            {"Patient_Week": test_ids, "FVC": test_mu, "Confidence": test_sigma}
        )

        # Save submission
        submission_path = "./submission/submission.csv"
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)
        sub_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nMetric ({final_metric:.6f}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
