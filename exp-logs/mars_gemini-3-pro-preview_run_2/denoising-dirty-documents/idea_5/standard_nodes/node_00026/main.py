import os
import torch
import numpy as np
from scipy.stats import pearsonr
import warnings

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import TextDenoisingDataset
from library.model import CACResUNet
from library.train import run_training
from library.inference import predict_tiled, run_inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # --- 1. Setup ---
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Define paths
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")
    os.makedirs(submission_dir, exist_ok=True)

    print("Starting Runfile Execution...")

    # --- 2. Training ---
    # Running a fast baseline with 30 epochs.
    # The default is 100, but 30 is sufficient for a strong baseline on this small dataset
    # and ensures we finish well within the time limit.
    print("\n--- Phase 1: Training ---")
    run_training(debug=False, num_epochs=30, patience=10, load_cached_data=True)

    # --- 3. Validation & Metric Calculation ---
    print("\n--- Phase 2: Validation & Evaluation ---")

    # Load the best model
    model = CACResUNet().to(device)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("Best model loaded for validation.")
    else:
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_SAVE_PATH}")

    model.eval()

    # Load Validation Dataset
    val_dataset = TextDenoisingDataset(
        metadata_path=Config.VAL_METADATA, mode="val", load_cached_data=True
    )

    # Accumulators for Global RMSE
    total_squared_error = 0.0
    total_pixels = 0

    # Accumulators for Failure Analysis (sampling to avoid OOM on analysis)
    # We will sample a percentage of pixels for correlation analysis
    analysis_errors = []
    analysis_inputs = []
    sample_rate = 0.1

    print(f"Validating on {len(val_dataset)} images...")

    with torch.no_grad():
        for i in range(len(val_dataset)):
            # Dataset returns: tensor_noisy (1, H, W), tensor_clean (1, H, W), img_id
            noisy_tensor, clean_tensor, img_id = val_dataset[i]

            # Move to device (add batch dim: 1, 1, H, W)
            noisy_input = noisy_tensor.unsqueeze(0).to(device)
            clean_target = clean_tensor.unsqueeze(0).to(device)

            # Predict Noise Residual
            pred_noise = predict_tiled(
                model,
                noisy_input,
                patch_size=Config.PATCH_SIZE,
                overlap=Config.TILE_OVERLAP,
            )

            # Reconstruct Clean Image
            pred_clean = noisy_input - pred_noise
            pred_clean = torch.clamp(pred_clean, 0.0, 1.0)

            # Calculate Error
            # Keep on GPU for calculation
            diff = clean_target - pred_clean
            squared_diff = diff**2

            total_squared_error += squared_diff.sum().item()
            total_pixels += clean_target.numel()

            # Collect data for failure analysis
            # Flatten and sample
            if np.random.rand() < 0.5:  # Process 50% of images for analysis stats
                flat_diff = diff.abs().cpu().numpy().flatten()
                flat_input = noisy_input.cpu().numpy().flatten()

                # Subsample pixels to keep memory usage low
                mask = np.random.rand(len(flat_diff)) < sample_rate
                analysis_errors.append(flat_diff[mask])
                analysis_inputs.append(flat_input[mask])

    # Compute Global RMSE
    final_rmse = np.sqrt(total_squared_error / total_pixels)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_rmse}")

    # --- 4. Failure Analysis ---
    print("\n--- Phase 3: Failure Analysis ---")
    if analysis_errors:
        all_errors = np.concatenate(analysis_errors)
        all_inputs = np.concatenate(analysis_inputs)

        # Correlation between Input Intensity and Error Magnitude
        # High intensity = White background, Low intensity = Black text
        corr, _ = pearsonr(all_inputs, all_errors)
        print(f"Correlation (Input Intensity vs Error Magnitude): {corr:.4f}")

        if corr < 0:
            print(
                "Observation: Errors are negatively correlated with intensity (Model struggles more with dark text regions)."
            )
        else:
            print(
                "Observation: Errors are positively correlated with intensity (Model struggles more with bright background regions)."
            )
    else:
        print("Insufficient data for failure analysis.")

    # --- 5. Submission ---
    print("\n--- Phase 4: Submission Generation ---")
    threshold = 0.009138691164531186

    if final_rmse < threshold:
        print(
            f"Validation metric ({final_rmse}) meets threshold ({threshold}). Generating submission..."
        )
        run_inference(
            model_path=Config.MODEL_SAVE_PATH,
            output_path=submission_path,
            batch_size=1,  # Inference function handles batching internally via tiling
            load_cached_data=True,
        )
        print(f"Submission saved to {submission_path}")
    else:
        print(
            f"Validation metric ({final_rmse}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
