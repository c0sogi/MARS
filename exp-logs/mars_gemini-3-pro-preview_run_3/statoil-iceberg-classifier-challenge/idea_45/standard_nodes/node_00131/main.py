import sys
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.dataset import get_loaders, get_test_loader
from library.engine import train_fold, generate_submission
from library.model import BDPH_CNN


def main():
    # 1. Setup Environment
    Config.setup()
    seed_everything()

    # Check device
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Cross-Validation Loop
    print(f"Starting {Config.N_FOLDS}-Fold Cross-Validation...")

    # Containers for Out-Of-Fold (OOF) data aggregation
    oof_preds = []
    oof_targets = []

    # Containers for Failure Analysis features
    oof_angles = []
    oof_b1_means = []
    oof_b1_stds = []

    for fold in range(Config.N_FOLDS):
        # Retrieve data loaders for the current fold
        # load_cached_data=True ensures we use pre-processed numpy arrays for speed
        train_loader, val_loader = get_loaders(fold, load_cached_data=True)

        # Train the model for this fold
        # train_fold handles the training loop, validation, and checkpoint saving
        train_fold(fold, train_loader, val_loader)

        # --- Validation Inference for OOF Analysis ---
        # Load the best model saved during training for this fold
        model = BDPH_CNN().to(device)
        checkpoint_path = os.path.join(
            Config.CHECKPOINT_DIR, f"model_best_fold_{fold}.pth"
        )

        # Load weights
        load_checkpoint(model, checkpoint_path)
        model.eval()

        fold_preds = []
        fold_targets = []

        # Disable gradient calculation for inference speed
        with torch.no_grad():
            for images, angles, targets in val_loader:
                images = images.to(device)
                angles = angles.to(device)

                # Forward pass
                outputs = model(images, angles)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()

                fold_preds.extend(probs)
                fold_targets.extend(targets.numpy().flatten())

                # Collect features for failure analysis
                # images shape: [B, 3, 75, 75]. Channel 0 is Band 1 (HH).
                b1 = images[:, 0, :, :]

                # Store metadata
                oof_angles.extend(angles.cpu().numpy().flatten())
                oof_b1_means.extend(b1.mean(dim=(1, 2)).cpu().numpy().flatten())
                oof_b1_stds.extend(b1.std(dim=(1, 2)).cpu().numpy().flatten())

        oof_preds.extend(fold_preds)
        oof_targets.extend(fold_targets)

    # 3. Final Evaluation
    oof_preds = np.array(oof_preds)
    oof_targets = np.array(oof_targets)

    # Calculate Log Loss
    final_metric = log_loss(oof_targets, oof_preds, labels=[0, 1])
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude
    errors = np.abs(oof_targets - oof_preds)

    # Create a DataFrame to analyze correlations
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": oof_angles,
            "b1_mean": oof_b1_means,
            "b1_std": oof_b1_stds,
        }
    )

    # Compute correlations
    correlations = df_analysis.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # 5. Submission Generation
    THRESHOLD = 0.1806015565870406

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) meets the threshold ({THRESHOLD}). Generating submission..."
        )
        # Get test loader
        test_loader = get_test_loader(load_cached_data=True)
        # Generate submission using the ensemble of trained models
        generate_submission(test_loader)
    else:
        print(
            f"\nMetric ({final_metric}) does not meet the threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
