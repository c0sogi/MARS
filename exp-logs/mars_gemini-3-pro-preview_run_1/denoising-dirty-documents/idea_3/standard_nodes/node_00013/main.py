import os
import numpy as np
import torch
import pandas as pd

# Import from provided library files
from library.config import Config, seed_everything
from library.train import run_training
from library.inference import generate_submission, predict_with_tta
from library.dataset import DenoisingDataset
from library.model import AttentionUNet
from library.utils import calculate_rmse


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override Config for a fast baseline execution
    Config.NUM_EPOCHS = 50  # Reduced from 1000 for speed
    Config.NUM_MODELS = 3  # Reduced from 5 to save time while keeping ensemble

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # =========================================================================
    # 2. Training
    # =========================================================================
    print("Starting ensemble training...")
    # run_training handles the loop over NUM_MODELS and saves checkpoints
    run_training(debug=False)

    # =========================================================================
    # 3. Validation & Failure Analysis
    # =========================================================================
    print("Starting validation and failure analysis...")

    # Load the validation dataset
    val_dataset = DenoisingDataset(split="val", debug=False)

    # Load the trained ensemble models
    models = []
    for i in range(Config.NUM_MODELS):
        model_path = os.path.join(Config.WORKING_DIR, f"model_{i}.pth")
        if os.path.exists(model_path):
            try:
                model = UNet(n_channels=1, n_classes=1).to(device)
                checkpoint = torch.load(model_path, map_location=device)
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                model.eval()
                models.append(model)
                print(f"Loaded model {i}")
            except Exception as e:
                print(f"Failed to load model {i}: {e}")

    if not models:
        raise RuntimeError("No models available for validation.")

    # Variables for metric calculation
    total_sq_error = 0.0
    total_pixels = 0

    # Variables for failure analysis
    img_rmses = []
    img_means = []
    img_stds = []

    # Inference loop on validation set
    for idx in range(len(val_dataset)):
        # val_dataset returns (noisy_tensor, clean_tensor)
        # Tensors are (1, H, W)
        noisy_t, clean_t = val_dataset[idx]

        # Collect input features for failure analysis
        noisy_np = noisy_t.numpy()
        img_means.append(np.mean(noisy_np))
        img_stds.append(np.std(noisy_np))

        # Prepare input for model: (1, 1, H, W)
        input_tensor = noisy_t.unsqueeze(0).to(device)

        # Ensemble Prediction with TTA
        ensemble_accum = None
        with torch.no_grad():
            for model in models:
                pred = predict_with_tta(model, input_tensor, device)
                if ensemble_accum is None:
                    ensemble_accum = pred
                else:
                    ensemble_accum += pred

        # Average predictions
        avg_pred = ensemble_accum / len(models)

        # Convert to numpy for metric calc
        pred_np = avg_pred.squeeze().cpu().numpy()
        clean_np = clean_t.squeeze().numpy()

        # Calculate errors
        diff = clean_np - pred_np
        sq_diff = diff**2

        # Accumulate global stats
        total_sq_error += np.sum(sq_diff)
        total_pixels += clean_np.size

        # Store per-image RMSE
        img_rmses.append(np.sqrt(np.mean(sq_diff)))

    # Compute Global RMSE
    global_mse = total_sq_error / total_pixels
    final_metric = np.sqrt(global_mse)

    # Print required metric
    print(f"Final Validation Metric: {final_metric:.20f}")

    # Perform Failure Analysis
    rmse_arr = np.array(img_rmses)
    mean_arr = np.array(img_means)
    std_arr = np.array(img_stds)

    # Calculate correlations
    # Handle edge case of zero variance if it occurs (unlikely with real images)
    if np.std(mean_arr) > 0:
        corr_mean = np.corrcoef(rmse_arr, mean_arr)[0, 1]
    else:
        corr_mean = 0.0

    if np.std(std_arr) > 0:
        corr_std = np.corrcoef(rmse_arr, std_arr)[0, 1]
    else:
        corr_std = 0.0

    print("-" * 30)
    print("Failure Analysis Report")
    print("-" * 30)
    print(f"Correlation (Error vs Input Mean Intensity): {corr_mean:.4f}")
    print(f"Correlation (Error vs Input Std Dev): {corr_std:.4f}")
    print("-" * 30)

    # =========================================================================
    # 4. Submission
    # =========================================================================
    THRESHOLD = 0.015272615302544418

    if final_metric < THRESHOLD:
        print(
            f"Validation metric {final_metric:.6f} passed threshold {THRESHOLD:.6f}. Generating submission..."
        )
        generate_submission(debug=False)
    else:
        print(
            f"Validation metric {final_metric:.6f} did not meet threshold {THRESHOLD:.6f}. Submission skipped."
        )


if __name__ == "__main__":
    main()
