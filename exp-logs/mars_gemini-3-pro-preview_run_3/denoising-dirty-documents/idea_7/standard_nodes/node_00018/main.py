import os
import torch
import numpy as np
from library.config import Config
from library.utils import set_seed
from library.train import run_training
from library.inference import generate_submission_file
from library.data_loader import get_dataloaders


def main():
    # 1. Setup and Reproducibility
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Initializing run on device: {device}")

    # 2. Fast Baseline Training
    # We limit max_epochs and max_batches_per_epoch to ensure a quick execution
    # while still allowing the model to learn from a representative subset of data.
    print("Starting fast baseline training...")
    model = run_training(load_cached_data=True, max_epochs=5, max_batches_per_epoch=500)

    # 3. Full Validation and Failure Analysis
    # We must evaluate on the entire validation set to get the official metric.
    print("Starting full validation and failure analysis...")
    _, val_loader = get_dataloaders(load_cached_data=True)

    model.eval()

    total_sse = 0.0
    total_pixels = 0

    # Accumulators for failure analysis
    # We collect flattened arrays of input intensities and error magnitudes
    all_errors = []
    all_inputs = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass: Model predicts the clean image directly
            clean_pred = model(inputs)

            # Reconstruct ground truth clean image
            # The loader provides noise_target = noisy_input - clean_ground_truth
            # Therefore, clean_ground_truth = noisy_input - noise_target
            clean_target = inputs - targets

            # Calculate pixel-wise difference
            diff = clean_pred - clean_target

            # Update Global RMSE stats
            # We sum the squared errors across all pixels in the batch
            squared_diff = diff**2
            total_sse += squared_diff.sum().item()
            total_pixels += squared_diff.numel()

            # Collect data for failure analysis
            # We use absolute error (L1) for correlation analysis
            error_magnitude = torch.abs(diff).cpu().numpy().flatten()
            input_intensity = inputs.cpu().numpy().flatten()

            all_errors.append(error_magnitude)
            all_inputs.append(input_intensity)

    # 4. Compute Final Metric
    if total_pixels > 0:
        final_mse = total_sse / total_pixels
        final_rmse = np.sqrt(final_mse)
    else:
        final_rmse = float("inf")

    # Print the required metric string
    print(f"Final Validation Metric: {final_rmse}")

    # 5. Perform Failure Analysis
    if all_errors:
        flat_errors = np.concatenate(all_errors)
        flat_inputs = np.concatenate(all_inputs)

        # Calculate Pearson correlation between input intensity and error magnitude
        # np.corrcoef returns the correlation matrix
        correlation = np.corrcoef(flat_inputs, flat_errors)[0, 1]
        print(f"Correlation between Input Intensity and Error Magnitude: {correlation}")
    else:
        print("Warning: No validation data available for failure analysis.")

    # 6. Conditional Submission Generation
    # Threshold defined in the task description
    THRESHOLD = 0.011577641381826402

    if final_rmse < THRESHOLD:
        print(
            f"Validation metric {final_rmse} meets threshold {THRESHOLD}. Generating submission..."
        )
        submission_path = Config.SUBMISSION_PATH
        model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

        generate_submission_file(
            model_path=model_path, output_path=submission_path, device=Config.DEVICE
        )
        print(f"Submission saved to {submission_path}")
    else:
        print(
            f"Validation metric {final_rmse} does not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
