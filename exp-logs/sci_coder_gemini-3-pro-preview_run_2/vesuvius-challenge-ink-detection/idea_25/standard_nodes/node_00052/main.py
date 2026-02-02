import os
import sys
import torch
import numpy as np
import shutil
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.data import InkDataset
from library.model import UnifiedSegFormer
from library.engine import Trainer
from library.utils import fbeta_score
from library.inference import run_inference, set_seed


def main():
    # 1. Setup and Reproducibility
    set_seed(Config.SEED)

    # Configure for Fast Baseline execution
    # We override Config attributes to ensure the script runs within the time limit
    # while still performing a meaningful training loop.
    Config.EPOCHS = 3
    Config.MAX_SAMPLES = 300  # Limit training data for speed

    print(
        f"Configuration: Epochs={Config.EPOCHS}, Max Samples={Config.MAX_SAMPLES}, Device={Config.DEVICE}"
    )

    # 2. Data Loading
    print("Initializing Data Loaders...")
    # Train dataset with limited samples
    train_dataset = InkDataset(
        mode="train", limit_size=Config.MAX_SAMPLES, load_cached_data=True
    )
    # Validation dataset (use full or reasonable subset for accurate metric)
    # We use the full validation set defined in metadata to get a proper score
    val_dataset = InkDataset(mode="validation", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = UnifiedSegFormer()
    model.to(Config.DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Training Loop
    print("Starting Training...")
    trainer = Trainer(model, train_loader, val_loader, optimizer, Config.DEVICE)
    trainer.fit(Config.EPOCHS)

    # 5. Final Validation and Failure Analysis
    print("Running Final Validation and Failure Analysis...")

    # Load the best model saved during training
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Loading best model from {Config.BEST_MODEL_PATH}")
        model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
        )
    else:
        print("Warning: Best model checkpoint not found. Using current model weights.")

    model.eval()

    running_score = 0.0
    total_samples = 0

    # Lists for failure analysis
    error_magnitudes = []
    input_intensities = []

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(Config.DEVICE, dtype=torch.float32)
            masks = masks.to(Config.DEVICE, dtype=torch.float32)
            batch_size = images.size(0)

            # Inference
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            # Metric Calculation (F0.5)
            # fbeta_score returns the score for the batch
            batch_score = fbeta_score(
                outputs, masks, beta=0.5, threshold=Config.BINARIZATION_THRESHOLD
            )
            running_score += batch_score * batch_size
            total_samples += batch_size

            # Failure Analysis Data Collection
            # Feature: Mean intensity of the input image (proxy for substrate density/noise)
            # Error: Mean Absolute Error (L1 distance) between probability and binary mask

            # Calculate mean intensity per image in the batch: (B, 3, H, W) -> (B,)
            batch_intensities = images.mean(dim=(1, 2, 3)).cpu().numpy()

            # Calculate MAE per image in the batch: (B, 1, H, W) -> (B,)
            batch_errors = torch.abs(probs - masks).mean(dim=(1, 2, 3)).cpu().numpy()

            input_intensities.extend(batch_intensities)
            error_magnitudes.extend(batch_errors)

    # Compute Final Metric
    final_metric = running_score / total_samples if total_samples > 0 else 0.0

    # Print required metric format
    print(f"Final Validation Metric: {final_metric}")

    # Compute Correlation for Failure Analysis
    if len(error_magnitudes) > 1:
        # Using numpy for correlation to avoid extra dependencies, though scipy is available
        # np.corrcoef returns a matrix [[1, r], [r, 1]]
        correlation = np.corrcoef(error_magnitudes, input_intensities)[0, 1]
        print(
            f"Failure Analysis: Correlation between Input Intensity and Error Magnitude: {correlation:.8f}"
        )
    else:
        print("Failure Analysis: Insufficient data for correlation.")

    # 6. Submission Generation
    # Threshold defined in task
    SUBMISSION_THRESHOLD = 0.597622633

    if final_metric > SUBMISSION_THRESHOLD:
        print(
            f"Validation metric ({final_metric}) exceeds threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )

        # Run inference using the provided library function
        # This handles loading the model, processing test data, and saving submission.csv
        run_inference(
            checkpoint_path=Config.BEST_MODEL_PATH,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
        )

        # Ensure the file is in the expected location if needed.
        # library.config writes to ./submission.csv.
        # If a specific subdirectory is required by the environment, we could move it here,
        # but we adhere to the library's configuration for the home directory output.
        if os.path.exists(Config.SUBMISSION_PATH):
            print(f"Submission successfully generated at {Config.SUBMISSION_PATH}")
        else:
            print("Error: Submission file was not created.")

    else:
        print(
            f"Validation metric ({final_metric}) did not exceed threshold ({SUBMISSION_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
