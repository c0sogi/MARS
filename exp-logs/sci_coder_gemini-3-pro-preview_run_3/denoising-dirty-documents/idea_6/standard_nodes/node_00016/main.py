import os
import sys
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
import warnings

# Import provided library modules
from library import config
from library import dataset
from library import train
from library import predict

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # 1. Setup & Configuration
    # Set seed for reproducibility
    config.set_seed(42)

    # Override config for Fast Baseline execution
    # Reduce epochs to ensure completion within time limit
    config.NUM_EPOCHS = 5
    # Increase stride to reduce the number of training patches (approx. 4x fewer samples)
    # This speeds up data generation and training significantly
    config.STRIDE = 20

    # 2. Training
    print("Starting training with fast baseline configuration...")
    # load_cached_data=False is crucial here to force regeneration of data
    # with the new STRIDE parameter.
    rdn_model = train.train_model(load_cached_data=False)

    # 3. Model Loading for Evaluation
    # train_model returns the model state at the last epoch.
    # We want to evaluate the BEST model saved during training.
    device = config.DEVICE
    best_model_path = config.MODEL_SAVE_PATH

    if os.path.exists(best_model_path):
        rdn_model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model weights not found. Using last epoch weights.")

    rdn_model.eval()

    # 4. Validation & Metric Calculation
    # Load validation data. We use load_cached_data=True because train_model
    # just generated the cache with the correct STRIDE.
    _, val_dataset = dataset.prepare_datasets(load_cached_data=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    criterion = nn.MSELoss()

    # Compute metrics on the full validation set
    _, rmse = train.validate(rdn_model, val_loader, criterion, device)

    # Print the required metric with full precision
    print(f"Final Validation Metric: {rmse:.16f}")

    # 5. Failure Analysis
    print("Performing failure analysis...")

    input_intensities = []
    error_magnitudes = []

    # We limit the number of samples for correlation analysis to prevent OOM
    # if the validation set is very large, though with stride 20 it should be manageable.
    MAX_SAMPLES = 100000
    current_samples = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = rdn_model(inputs)

            # Calculate absolute error per pixel
            # Error = |Predicted - Target|
            errors = torch.abs(outputs - targets)

            # Flatten and collect data
            batch_intensities = inputs.cpu().numpy().flatten()
            batch_errors = errors.cpu().numpy().flatten()

            input_intensities.append(batch_intensities)
            error_magnitudes.append(batch_errors)

            current_samples += len(batch_intensities)
            if current_samples >= MAX_SAMPLES:
                break

    # Concatenate collected arrays
    flat_intensities = np.concatenate(input_intensities)[:MAX_SAMPLES]
    flat_errors = np.concatenate(error_magnitudes)[:MAX_SAMPLES]

    # Calculate Pearson correlation
    corr, _ = pearsonr(flat_intensities, flat_errors)
    print(f"Correlation between Input Intensity and Error Magnitude: {corr:.16f}")

    # 6. Submission Generation
    THRESHOLD = 0.011577641381826402

    if rmse < THRESHOLD:
        print(
            f"Validation metric {rmse} is lower than threshold {THRESHOLD}. Generating submission..."
        )
        predict.generate_predictions(
            model_path=best_model_path,
            metadata_path=config.TEST_METADATA_PATH,
            output_path=config.SUBMISSION_PATH,
            device=device,
        )
    else:
        print(
            f"Validation metric {rmse} is NOT lower than threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
