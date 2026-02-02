import sys
import os
import torch
import numpy as np
from library.config import GlobalConfig, StreamAConfig, StreamBConfig
from library.utils import seed_everything
from library.train import train_model
from library.inference import generate_submission, load_models
from library.dataset import get_dataloader


def main():
    # 1. Setup
    seed_everything(GlobalConfig.SEED)
    print("Starting Heterogeneous Resolution-Capacity Ensemble Pipeline...")

    # 2. Train Stream A Models (Context Specialists)
    # Using reduced epochs for fast baseline execution as per requirements
    FAST_EPOCHS = 5

    print(f"\n--- Training Stream A ({StreamAConfig.NAME}) ---")
    for seed in GlobalConfig.STREAM_A_SEEDS:
        train_model(StreamAConfig, seed, epochs=FAST_EPOCHS, debug=False)

    # 3. Train Stream B Models (Diversity Specialists)
    print(f"\n--- Training Stream B ({StreamBConfig.NAME}) ---")
    for seed in GlobalConfig.STREAM_B_SEEDS:
        train_model(StreamBConfig, seed, epochs=FAST_EPOCHS, debug=False)

    # 4. Validation Assessment
    print("\n--- Starting Validation Assessment ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the ensemble
    models = load_models(device)
    if not models:
        print("Error: No models loaded. Exiting.")
        return

    # Get validation loader
    val_loader = get_dataloader(mode="val", shuffle=False)

    total_squared_error = 0.0
    total_pixels = 0

    # Storage for failure analysis
    img_errors = []
    meta_means = []
    meta_areas = []

    # Criterion for accumulation (Sum of Squared Errors)
    criterion = torch.nn.MSELoss(reduction="sum")

    with torch.no_grad():
        for i, (noisy, clean, _) in enumerate(val_loader):
            noisy = noisy.to(device)
            clean = clean.to(device)

            # Ensemble Inference (Simple Averaging for Validation Speed)
            batch_accum = torch.zeros_like(noisy)
            for model in models:
                batch_accum += model(noisy)

            pred = batch_accum / len(models)

            # Calculate Error
            loss = criterion(pred, clean)
            sse = loss.item()
            n_pixels = clean.numel()

            total_squared_error += sse
            total_pixels += n_pixels

            # Per-image metrics for analysis
            rmse_img = np.sqrt(sse / n_pixels)
            img_errors.append(rmse_img)

            # Extract features (on CPU)
            noisy_np = noisy.cpu().numpy()
            meta_means.append(np.mean(noisy_np))
            meta_areas.append(noisy_np.size)

    # Compute Global Metric
    if total_pixels > 0:
        global_mse = total_squared_error / total_pixels
        final_rmse = np.sqrt(global_mse)
    else:
        final_rmse = float("inf")

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_rmse}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    if len(img_errors) > 1:
        # Correlation with Input Mean Intensity
        corr_mean = np.corrcoef(img_errors, meta_means)[0, 1]
        # Correlation with Input Image Area
        corr_area = np.corrcoef(img_errors, meta_areas)[0, 1]

        print(f"Correlation (Error vs Input Mean Intensity): {corr_mean:.4f}")
        print(f"Correlation (Error vs Input Image Area): {corr_area:.4f}")
    else:
        print("Insufficient validation samples for correlation analysis.")

    # 6. Submission Generation
    # Threshold defined in task description
    SUBMISSION_THRESHOLD = 0.011870221132053216

    if final_rmse < SUBMISSION_THRESHOLD:
        print(
            f"\nValidation metric ({final_rmse}) meets threshold ({SUBMISSION_THRESHOLD})."
        )
        print("Generating submission file...")
        generate_submission(load_cached_data=True)
    else:
        print(
            f"\nValidation metric ({final_rmse}) does not meet threshold ({SUBMISSION_THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
