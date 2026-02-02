import os
import torch
import numpy as np
import pandas as pd
import scipy.stats as stats

# Import components from the provided library
from library.config import Config
from library.utils import seed_everything, inverse_scale, laplace_log_likelihood
from library.data import get_dataloaders
from library.model import BCOSRNet
from library.engine import run_training, generate_submission


def main():
    # 1. Initialization and Configuration
    print("Initializing Fast Baseline Run...")
    Config.setup()

    # Override Config for a fast baseline execution
    Config.EPOCHS = 30  # Increased to allow convergence with stable loss
    Config.T_MAX = 30  # Link scheduler horizon (Cite Lesson 100)
    Config.BATCH_SIZE = 32

    # Set device
    if torch.cuda.is_available():
        Config.DEVICE = "cuda"
        print("CUDA is available. Using GPU.")
    else:
        Config.DEVICE = "cpu"
        print("CUDA not available. Using CPU.")

    # 2. Training
    print("\n--- Starting Training ---")
    # run_training handles the training loop, validation monitoring, and saving the best model
    _ = run_training(debug=False)

    # 3. Load Best Model for Analysis
    print("\n--- Loading Best Model for Evaluation ---")
    device = torch.device(Config.DEVICE)
    model = BCOSRNet()
    checkpoint_path = os.path.join(Config.CHECKPOINTS_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print("Error: Checkpoint not found! Training may have failed.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # 4. Validation Inference & Metric Calculation
    print("\n--- Performing Validation Inference ---")
    _, val_loader = get_dataloaders(debug=False)

    all_targets = []
    all_pred_mean = []
    all_pred_sigma = []

    # Disable gradients for inference
    with torch.no_grad():
        for images, clinical, targets in val_loader:
            images = images.to(device)
            clinical = clinical.to(device)
            targets = targets.to(device)

            # Forward pass
            preds = model(images, clinical)
            pred_mean_norm = preds[:, 0]
            pred_sigma_norm = preds[:, 1]

            # Inverse Scale to get ML units
            pred_mean_abs, pred_sigma_abs = inverse_scale(
                pred_mean_norm, pred_sigma_norm
            )
            target_abs = targets * Config.TARGET_STD + Config.TARGET_MEAN

            # Collect results
            all_targets.extend(target_abs.cpu().numpy())
            all_pred_mean.extend(pred_mean_abs.cpu().numpy())
            all_pred_sigma.extend(pred_sigma_abs.cpu().numpy())

    # Convert to numpy arrays
    all_targets = np.array(all_targets)
    all_pred_mean = np.array(all_pred_mean)
    all_pred_sigma = np.array(all_pred_sigma)

    # Compute Final Metric
    final_metric = laplace_log_likelihood(
        all_targets, all_pred_mean, all_pred_sigma, clip_sigma=True, clip_delta=True
    )

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Load validation metadata. Since val_loader has shuffle=False, rows align.
    val_df = pd.read_csv(Config.VAL_CSV)

    # Calculate Absolute Error
    errors = np.abs(all_targets - all_pred_mean)

    if len(val_df) == len(errors):
        val_df["Error"] = errors

        # Numerical Features to analyze
        features = ["Weeks", "Percent", "Age"]

        print("Correlation between Absolute Error and Features:")
        for feat in features:
            if feat in val_df.columns:
                corr, _ = stats.pearsonr(val_df[feat], val_df["Error"])
                print(f"  {feat}: {corr:.4f}")
            else:
                print(f"  {feat}: Not found in metadata")

        # Categorical Analysis
        print("Mean Error by Category:")
        for cat in ["Sex", "SmokingStatus"]:
            if cat in val_df.columns:
                print(f"  {cat}:")
                print(val_df.groupby(cat)["Error"].mean())
    else:
        print(
            f"Warning: Validation set size ({len(val_df)}) matches predictions ({len(errors)}) mismatch."
        )

    # 6. Submission
    THRESHOLD = -6.573619738753321

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
