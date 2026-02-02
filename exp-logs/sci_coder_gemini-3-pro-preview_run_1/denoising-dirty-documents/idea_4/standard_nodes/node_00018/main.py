import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import DeepSupervisionUNet
from library.train import train_model
from library.inference import predict_with_ensemble, create_submission_file


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Setup directories
    Config.setup()

    # Set random seeds for reproducibility
    seed_everything(Config.SEED)

    print("=== Starting Orchestration ===")
    print(f"Configuration: {Config.NUM_EPOCHS} Epochs, {Config.NUM_MODELS} Model(s)")
    print(f"Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------
    print("\n--- Phase 1: Training ---")
    # Train the ensemble (or single model)
    # We use load_cached_data=True to utilize any pre-processed .npz files
    for i in range(Config.NUM_MODELS):
        train_model(model_index=i, load_cached_data=True)

    # -------------------------------------------------------------------------
    # 3. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Phase 2: Validation & Failure Analysis ---")

    device = torch.device(Config.DEVICE)

    # Load validation data
    # get_dataloaders returns (train, val, test). We only need val here.
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Load trained models for inference
    models = []
    for i in range(Config.NUM_MODELS):
        model_path = os.path.join(Config.WORKING_DIR, f"model_{i}.pth")
        if os.path.exists(model_path):
            model = DeepSupervisionUNet()
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            models.append(model)
        else:
            print(f"Warning: Model {i} checkpoint not found.")

    if not models:
        print("Error: No models available for validation.")
        return

    # Containers for metrics and analysis
    val_stats = []
    total_squared_error = 0.0
    total_pixels = 0

    print("Running validation inference...")

    with torch.no_grad():
        for batch_idx, (noisy, clean) in enumerate(val_loader):
            noisy = noisy.to(device)
            clean = clean.to(device)

            # Ensemble Prediction (Simple averaging, no TTA for speed in validation)
            preds = []
            for model in models:
                # Model output shape: (1, 1, H, W)
                out = model(noisy)
                preds.append(out)

            # Average predictions across ensemble
            avg_pred = torch.mean(torch.stack(preds), dim=0)

            # Clamp to valid range [0, 1]
            avg_pred = torch.clamp(avg_pred, 0.0, 1.0)

            # Convert to numpy for metric calculation
            # Shapes: (1, 1, H, W) -> (H, W)
            p_np = avg_pred.squeeze().cpu().numpy()
            c_np = clean.squeeze().cpu().numpy()
            n_np = noisy.squeeze().cpu().numpy()

            # 1. Calculate Squared Error for Global RMSE
            diff = p_np - c_np
            squared_error = np.sum(diff**2)
            num_px = p_np.size

            total_squared_error += squared_error
            total_pixels += num_px

            # 2. Calculate Image-wise RMSE for Failure Analysis
            img_mse = np.mean(diff**2)
            img_rmse = np.sqrt(img_mse)

            # 3. Extract Features for Failure Analysis
            mean_intensity = np.mean(n_np)
            std_intensity = np.std(n_np)
            area = num_px

            val_stats.append(
                {
                    "mean_intensity": mean_intensity,
                    "std_intensity": std_intensity,
                    "area": area,
                    "rmse": img_rmse,
                }
            )

    # Calculate Final Global RMSE
    # Metric: RMSE between cleaned pixel intensities and actual grayscale pixel intensities
    final_metric = np.sqrt(total_squared_error / total_pixels)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # Perform Failure Analysis
    print("\nFailure Analysis (Correlation with Error):")
    if len(val_stats) > 1:
        df_analysis = pd.DataFrame(val_stats)

        # Calculate correlations
        corr_mean = df_analysis["mean_intensity"].corr(df_analysis["rmse"])
        corr_std = df_analysis["std_intensity"].corr(df_analysis["rmse"])
        corr_area = df_analysis["area"].corr(df_analysis["rmse"])

        print(f"Correlation (Mean Intensity vs RMSE): {corr_mean:.4f}")
        print(f"Correlation (Std Intensity vs RMSE): {corr_std:.4f}")
        print(f"Correlation (Area vs RMSE): {corr_area:.4f}")
    else:
        print("Insufficient validation samples for correlation analysis.")

    # -------------------------------------------------------------------------
    # 4. Submission Generation
    # -------------------------------------------------------------------------
    print("\n--- Phase 3: Submission Check ---")

    THRESHOLD = 0.012221260240721992

    if final_metric < THRESHOLD:
        print(f"Metric {final_metric:.6f} is lower than threshold {THRESHOLD:.6f}.")
        print("Generating submission file...")

        # Generate predictions using the ensemble (includes TTA)
        predictions = predict_with_ensemble(load_cached_data=True)

        # Format and save submission
        create_submission_file(predictions)

    else:
        print(f"Metric {final_metric:.6f} did not meet threshold {THRESHOLD:.6f}.")
        print("Submission generation skipped.")


if __name__ == "__main__":
    main()
