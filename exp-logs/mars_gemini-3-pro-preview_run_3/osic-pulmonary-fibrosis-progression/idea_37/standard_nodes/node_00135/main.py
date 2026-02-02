import os
import sys
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data import get_dataloaders
from library.model import RCOSRNet
from library.train import Trainer
from library.inference import predict_test_set


def main():
    # 1. Setup and Configuration
    # Initialize environment and seeds
    seed_everything(Config.SEED)
    Config.setup()

    # Fast Baseline Overrides
    # Reducing epochs to ensure the script completes quickly (within ~2 hours limit)
    # while still allowing enough convergence for the small dataset.
    Config.EPOCHS = 15

    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    # load_cached_data=True is default in the library function, utilizing ./working cache
    train_loader, val_loader = get_dataloaders(debug=False)

    # 3. Model Initialization
    print("Initializing RCOSR-Net model...")
    model = RCOSRNet().to(device)

    # 4. Training
    # The Trainer class handles the training loop, validation per epoch, and checkpointing
    print(f"Starting training for {Config.EPOCHS} epochs...")
    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.fit(epochs=Config.EPOCHS)

    # 5. Validation and Failure Analysis
    print("\nPerforming final validation and failure analysis...")

    # Load the best model checkpoint
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    model.eval()

    # Containers for analysis
    all_trues = []
    all_preds = []
    all_sigmas = []
    all_features = []

    # Inference on Validation Set
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            clinical = batch["clinical"].to(device)
            raw_fvcs = batch["raw_fvc"].numpy()

            # Forward pass
            mu, sigma = model(imgs, clinical)

            # Inverse Transform
            mu_np = mu.cpu().numpy().flatten()
            sigma_np = sigma.cpu().numpy().flatten()

            # De-standardize to original scale
            mu_orig = mu_np * Config.TARGET_STD + Config.TARGET_MEAN
            sigma_orig = sigma_np * Config.TARGET_STD

            # Store results
            all_trues.extend(raw_fvcs)
            all_preds.extend(mu_orig)
            all_sigmas.extend(sigma_orig)

            # Store features for failure analysis
            # Clinical vector: [Base_FVC_Std, Time_Scaled, Age_Std, Sex, Smoke]
            all_features.append(clinical.cpu().numpy())

    # Calculate Final Metric
    final_metric = calculate_metric(all_trues, all_preds, all_sigmas)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Create a DataFrame to analyze correlations between Error and Features
    features_stack = np.vstack(all_features)
    analysis_df = pd.DataFrame(
        features_stack, columns=["Base_FVC", "Time", "Age", "Sex", "Smoke"]
    )

    # Calculate Absolute Error
    analysis_df["Abs_Error"] = np.abs(np.array(all_trues) - np.array(all_preds))

    # Compute correlations
    print("\nFailure Analysis - Correlation with Absolute Error:")
    correlations = analysis_df.corr()["Abs_Error"].sort_values(ascending=False)
    print(correlations)

    # 6. Submission Generation
    # Threshold defined in task
    THRESHOLD = -6.573619738753321

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        predict_test_set(model, device)
    else:
        print(
            f"\nMetric ({final_metric}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
