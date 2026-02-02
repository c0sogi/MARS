import sys
import os
import torch
import numpy as np
import warnings
import torch.nn.functional as F

# Suppress warnings
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config, seed_everything
from library.train import run_training
from library.predict import run_inference
from library.model import UNet
from library.dataset import get_dataloaders
from library.utils import load_checkpoint, calculate_rmse


def main():
    # 1. Setup Environment
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Define fast baseline parameters
    # Reducing epochs to 20 to ensure execution within time limits while allowing convergence on small data
    FAST_EPOCHS = 20

    # 2. Train the Model
    # We pass load_cached_data=True to utilize pre-processed .npy files if available
    run_training(num_epochs=FAST_EPOCHS, load_cached_data=True)

    # 3. Validation Assessment & Failure Analysis
    print("\n--- Starting Validation Assessment ---")

    # Initialize model
    model = UNet(n_channels=Config.IN_CHANNELS, n_classes=Config.OUT_CHANNELS).to(
        device
    )

    # Load the best checkpoint saved during training
    checkpoint = load_checkpoint(Config.MODEL_SAVE_PATH, model, device=device)
    if checkpoint is None:
        print("Error: No checkpoint found. Validation cannot proceed.")
        return

    model.eval()

    # Get Validation DataLoader
    # We use batch_size=1 because validation images have varying dimensions
    _, val_loader, _, _ = get_dataloaders(batch_size=1, load_cached_data=True)

    all_targets = []
    all_preds = []

    # Metrics for Failure Analysis
    img_error_magnitudes = []  # Mean Absolute Error per image
    img_input_means = []  # Mean intensity of input image
    img_input_stds = []  # Std dev of input image

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            # Keep targets on CPU for final metric calculation
            targets_np = targets.numpy()

            # Handle U-Net dimension requirements (divisible by 16)
            h, w = inputs.shape[2], inputs.shape[3]
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16

            if pad_h > 0 or pad_w > 0:
                inputs_padded = F.pad(inputs, (0, pad_w, 0, pad_h), mode="reflect")
            else:
                inputs_padded = inputs

            # Inference
            outputs_padded = model(inputs_padded)

            # Crop back to original size
            outputs = outputs_padded[:, :, :h, :w]
            outputs = torch.clamp(outputs, 0, 1)

            preds_np = outputs.cpu().numpy()

            # Store for global metric
            all_targets.append(targets_np)
            all_preds.append(preds_np)

            # --- Failure Analysis Data Collection ---
            # Calculate Mean Absolute Error for this specific image
            mae = np.mean(np.abs(preds_np - targets_np))
            img_error_magnitudes.append(mae)

            # Calculate Input Statistics
            input_np = inputs.cpu().numpy()
            img_input_means.append(np.mean(input_np))
            img_input_stds.append(np.std(input_np))

    # Compute Global RMSE (Final Validation Metric)
    # Concatenate all pixels to compute one metric over the entire dataset
    y_true_flat = np.concatenate([t.flatten() for t in all_targets])
    y_pred_flat = np.concatenate([p.flatten() for p in all_preds])

    final_rmse = calculate_rmse(y_true_flat, y_pred_flat)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_rmse}")

    # --- Failure Analysis Correlations ---
    print("\n--- Failure Analysis ---")
    if len(img_error_magnitudes) > 1:
        # Calculate Pearson correlation
        corr_mean = np.corrcoef(img_error_magnitudes, img_input_means)[0, 1]
        corr_std = np.corrcoef(img_error_magnitudes, img_input_stds)[0, 1]

        print(
            f"Correlation (Error Magnitude vs Input Mean Intensity): {corr_mean:.10f}"
        )
        print(f"Correlation (Error Magnitude vs Input Std Deviation): {corr_std:.10f}")
    else:
        print("Insufficient validation samples for correlation analysis.")

    # 4. Conditional Submission
    THRESHOLD = 0.016654925420880318

    if final_rmse < THRESHOLD:
        print(
            f"\nMetric {final_rmse} is lower than threshold {THRESHOLD}. Generating submission..."
        )
        run_inference(
            model_path=Config.MODEL_SAVE_PATH,
            output_path=Config.SUBMISSION_PATH,
            load_cached_data=True,
        )
    else:
        print(
            f"\nMetric {final_rmse} is not lower than threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
