import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch.nn.functional as F

from library.config import Config
from library.utils import set_seed, pad_image, unpad_image
from library.dataset import get_dataloaders
from library.model import UNet
from library.train import train_model
from library.predict import generate_submission


def main():
    # 1. Setup Environment
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Initializing workflow...")

    # 2. Train the Model
    # We use more epochs with augmentation and scheduling to improve performance.
    print("Step 1: Training Model...")
    train_model(epochs=Config.NUM_EPOCHS)

    # 3. Validation Assessment & Failure Analysis
    print("Step 2: Validation and Failure Analysis...")

    # Load the best model saved during training
    model = UNet(n_channels=Config.NUM_CHANNELS, n_classes=1, bilinear=True)
    model.to(device)

    if os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        state_dict = torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("Loaded best model checkpoint for validation.")
    else:
        print("Warning: Checkpoint not found. Using current model weights.")

    model.eval()

    # Get validation dataloader
    _, val_loader, _ = get_dataloaders()

    total_squared_error = 0.0
    total_pixels = 0

    img_rmses = []
    img_means = []
    img_stds = []

    # Inference loop for validation
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # TTA Inference
            original_shape = (inputs.shape[2], inputs.shape[3])
            padded_inputs = pad_image(inputs, divisor=32, mode="reflect")

            # 1. Original
            out1 = model(padded_inputs)
            # 2. Horizontal Flip
            out2 = torch.flip(model(torch.flip(padded_inputs, [3])), [3])
            # 3. Vertical Flip
            out3 = torch.flip(model(torch.flip(padded_inputs, [2])), [2])
            # 4. HV Flip
            out4 = torch.flip(model(torch.flip(padded_inputs, [2, 3])), [2, 3])

            avg_out = (out1 + out2 + out3 + out4) / 4.0
            outputs = unpad_image(avg_out, original_shape)

            # --- Global Metric Calculation ---
            # Calculate squared error for this batch
            # inputs/targets are normalized [0,1], so we compute MSE on that scale
            mse = F.mse_loss(outputs, targets, reduction="sum")
            total_squared_error += mse.item()
            total_pixels += targets.numel()

            # --- Failure Analysis Data Collection ---
            # Calculate per-image RMSE for correlation analysis
            # Batch size is 1 for validation
            img_mse = F.mse_loss(outputs, targets, reduction="mean").item()
            img_rmse = np.sqrt(img_mse)
            img_rmses.append(img_rmse)

            # Collect input features (Mean intensity and Std Dev)
            input_mean = inputs.mean().item()
            input_std = inputs.std().item()
            img_means.append(input_mean)
            img_stds.append(input_std)

    # Calculate Final Global RMSE
    global_mse = total_squared_error / total_pixels if total_pixels > 0 else 0.0
    final_metric = np.sqrt(global_mse)

    # Print required metric
    print(f"Final Validation Metric: {final_metric}")

    # Perform Failure Analysis
    if len(img_rmses) > 1:
        # Correlation between Input Mean Intensity and Error
        corr_mean, _ = pearsonr(img_means, img_rmses)
        # Correlation between Input Contrast (Std) and Error
        corr_std, _ = pearsonr(img_stds, img_rmses)

        print("-" * 30)
        print("Failure Analysis:")
        print(f"Correlation (Input Mean vs RMSE): {corr_mean:.4f}")
        print(f"Correlation (Input Std vs RMSE): {corr_std:.4f}")
        print("-" * 30)

        if abs(corr_mean) > 0.3:
            print(
                "Observation: Significant correlation found between image brightness and model error."
            )

    # 4. Generate Submission
    # We explicitly call the generate_submission from library.predict as it handles
    # padding/unpadding which is robust for test images of varying sizes.
    print("Step 3: Generating Submission...")

    threshold = 0.015272615302544418
    if final_metric < threshold:
        print(
            f"Validation metric {final_metric:.6f} is better than threshold {threshold:.6f}. Generating submission."
        )
        generate_submission()
    else:
        print(
            f"Validation metric {final_metric:.6f} did not improve upon threshold {threshold:.6f}. Skipping submission."
        )


if __name__ == "__main__":
    main()
