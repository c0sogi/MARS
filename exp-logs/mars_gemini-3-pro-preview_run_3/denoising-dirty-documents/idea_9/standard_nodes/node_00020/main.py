import sys
import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import components from the provided library
from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.data import get_dataloaders
from library.train import train_model
from library.model import ZIResDnCNN
from library.inference import run_inference


def main():
    # 1. Initialization
    # Set random seeds for reproducibility
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Training Phase
    # We limit the number of epochs to 5 to ensure the script completes quickly (Fast Baseline).
    # We use the full dataset to ensure the model learns representative features.
    print("--- Starting Training Phase ---")
    train_model(num_epochs=5, batch_size=Config.BATCH_SIZE, load_cached_data=True)

    # 3. Validation & Failure Analysis Phase
    print("\n--- Starting Validation & Failure Analysis ---")

    # Initialize the model architecture
    model = ZIResDnCNN(
        num_blocks=Config.NUM_BLOCKS,
        num_channels=Config.NUM_CHANNELS,
        kernel_size=Config.KERNEL_SIZE,
        padding=Config.PADDING,
        use_zero_gamma=Config.USE_ZERO_GAMMA,
    ).to(device)

    # Load the best checkpoint saved during the training phase
    if os.path.exists(Config.MODEL_SAVE_PATH):
        load_checkpoint(model, filename=Config.MODEL_SAVE_PATH)
        print(f"Loaded best model from {Config.MODEL_SAVE_PATH}")
    else:
        print(
            f"Warning: Checkpoint not found at {Config.MODEL_SAVE_PATH}. Analysis may be invalid."
        )

    # Set model to evaluation mode
    model.eval()

    # Retrieve validation dataloader
    # We reload the data here to ensure we have access to the raw inputs/targets for analysis
    _, val_loader = get_dataloaders(batch_size=Config.BATCH_SIZE, load_cached_data=True)

    # Containers for failure analysis
    all_preds = []
    all_targets = []
    all_inputs = []

    total_sse = 0.0
    total_pixels = 0

    # Inference loop (No Gradient Calculation for speed/memory)
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Clamp outputs to valid pixel range [0, 1]
            outputs = torch.clamp(outputs, 0, 1)

            # Calculate Sum of Squared Errors for this batch
            diff = outputs - targets
            batch_sse = torch.sum(diff**2).item()
            total_sse += batch_sse
            total_pixels += targets.numel()

            # Store batch results (move to CPU to free GPU memory)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_inputs.append(inputs.cpu().numpy())

    # Calculate Global RMSE
    mse = total_sse / total_pixels
    rmse = np.sqrt(mse)

    # Print the required validation metric
    print(f"Final Validation Metric: {rmse}")

    # --- Failure Analysis ---
    print("Performing failure analysis...")

    # Flatten arrays for correlation calculation
    flat_preds = np.concatenate(all_preds).flatten()
    flat_targets = np.concatenate(all_targets).flatten()
    flat_inputs = np.concatenate(all_inputs).flatten()

    # Calculate absolute error per pixel
    abs_errors = np.abs(flat_preds - flat_targets)

    # Correlation 1: Error vs Input Intensity
    # Helps identify if the model struggles with specific brightness levels
    corr_input, _ = pearsonr(abs_errors, flat_inputs)
    print(f"Correlation (Error vs Input Intensity): {corr_input}")

    # Correlation 2: Error vs Target Intensity
    # Helps identify if the model struggles with text (dark) or background (light)
    corr_target, _ = pearsonr(abs_errors, flat_targets)
    print(f"Correlation (Error vs Target Intensity): {corr_target}")

    # 4. Submission Logic
    # Strict threshold check as per requirements
    SUBMISSION_THRESHOLD = 0.011577641381826402

    if rmse < SUBMISSION_THRESHOLD:
        print(f"\nValidation RMSE ({rmse}) meets threshold ({SUBMISSION_THRESHOLD}).")
        print("Generating submission file...")

        run_inference(
            checkpoint_path=Config.MODEL_SAVE_PATH,
            output_path=Config.SUBMISSION_PATH,
            metadata_path=Config.TEST_METADATA_PATH,
            use_tta=Config.USE_TTA,
            device=Config.DEVICE,
        )
    else:
        print(
            f"\nValidation RMSE ({rmse}) did not meet threshold ({SUBMISSION_THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
