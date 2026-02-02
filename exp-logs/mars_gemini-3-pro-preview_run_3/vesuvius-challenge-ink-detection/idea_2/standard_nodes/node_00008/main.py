import sys
import os
import torch
import numpy as np
import warnings

# Import library modules
from library.config import Config, seed_everything
from library.train import train
from library.inference import generate_submission
from library.data import get_dataloaders
from library.model import LeanUNet25D
from library.utils import fbeta_score

# Suppress warnings for clean execution
warnings.filterwarnings("ignore")


def run_pipeline():
    # 1. Configuration and Setup
    # --------------------------
    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Configure for a fast baseline execution
    # Reducing epochs to 3 ensures the script completes well within the 2-hour limit
    # while providing enough iterations for the model to learn the task.
    Config.EPOCHS = 3

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print("Starting Ink Detection Pipeline...")

    # 2. Model Training
    # -----------------
    # Train the model using the configuration.
    # This saves the best EMA model to Config.MODEL_PATH.
    train(Config)

    # 3. Validation and Failure Analysis
    # ----------------------------------
    print("Starting Validation and Failure Analysis...")

    device = torch.device(Config.DEVICE)

    # Load the best trained model
    model = LeanUNet25D().to(device)
    if os.path.exists(Config.MODEL_PATH):
        state_dict = torch.load(Config.MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Error: Model checkpoint not found at {Config.MODEL_PATH}")
        return

    model.eval()

    # Get Validation DataLoader
    dataloaders = get_dataloaders(Config)
    val_loader = dataloaders.get("val")

    if val_loader is None:
        print("Error: Validation dataloader not found.")
        return

    # Containers for global metric calculation
    all_preds = []
    all_targets = []

    # Containers for failure analysis (subsampled)
    errors_sample = []
    intensities_sample = []

    # Inference loop
    with torch.no_grad():
        for batch in val_loader:
            # Unpack batch: images, labels, coordinates
            images, labels, _ = batch

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Move to CPU for analysis
            preds_np = outputs.cpu().numpy()  # Shape: (B, 1, H, W)
            labels_np = labels.cpu().numpy()  # Shape: (B, 1, H, W)

            # Store for metric calculation
            all_preds.append(preds_np)
            all_targets.append(labels_np)

            # --- Failure Analysis Data Collection ---
            # Calculate pixel-wise error magnitude
            # Squeeze channel dimension: (B, 1, H, W) -> (B, H, W)
            p_sq = preds_np.squeeze(1)
            l_sq = labels_np.squeeze(1)

            batch_error = np.abs(p_sq - l_sq)

            # Calculate input feature: Mean Pixel Intensity across Z-depth
            # images: (B, 65, H, W) -> mean -> (B, H, W)
            batch_intensity = images.mean(dim=1).cpu().numpy()

            # Flatten arrays for correlation
            error_flat = batch_error.flatten()
            intensity_flat = batch_intensity.flatten()

            # Subsample to save memory (1% of pixels)
            mask = np.random.rand(len(error_flat)) < 0.01

            errors_sample.extend(error_flat[mask])
            intensities_sample.extend(intensity_flat[mask])

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate Final Validation Metric (F0.5 Score)
    val_metric = fbeta_score(
        all_preds, all_targets, beta=0.5, threshold=Config.THRESHOLD
    )

    # Print metric in required format
    print(f"Final Validation Metric: {val_metric}")

    # Calculate Correlation for Failure Analysis
    if len(errors_sample) > 1:
        # Use numpy for Pearson correlation
        corr_matrix = np.corrcoef(errors_sample, intensities_sample)
        corr = corr_matrix[0, 1]
        print(
            f"Failure Analysis - Correlation between Error and Mean Pixel Intensity: {corr}"
        )
    else:
        print("Failure Analysis - Insufficient data for correlation.")

    # 4. Submission Generation
    # ------------------------
    # Threshold condition
    SUBMISSION_THRESHOLD = 0.38412588834762573

    if val_metric > SUBMISSION_THRESHOLD:
        print("Validation metric meets threshold. Generating submission...")
        generate_submission(
            checkpoint_path=Config.MODEL_PATH, output_path=Config.SUBMISSION_PATH
        )
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation metric ({val_metric}) did not exceed threshold ({SUBMISSION_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    run_pipeline()
