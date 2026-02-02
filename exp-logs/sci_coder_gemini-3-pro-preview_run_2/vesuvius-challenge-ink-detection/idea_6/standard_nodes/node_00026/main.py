import os
import numpy as np
import torch
import torch.nn.functional as F

from library.config import Config
from library.utils import setup_reproducibility, fbeta_score
from library.data import get_dataloaders
from library.model import get_model
from library.train import train_model
from library.inference import generate_submission


def main():
    # 1. Setup & Configuration
    # Ensure reproducibility
    setup_reproducibility(Config.SEED)

    # Modify Config for the task requirements
    # Force save the model by setting threshold to -1.0 so we can always evaluate/analyze it
    Config.BASELINE_SCORE_THRESHOLD = -1.0
    # Update submission path to match requirement "./submission/submission.csv"
    os.makedirs("./submission", exist_ok=True)
    Config.SUBMISSION_PATH = os.path.join("./submission", "submission.csv")
    # Limit epochs for fast baseline execution (dataset is small, so 15 is sufficient and fast)
    Config.EPOCHS = 15

    # 2. Training
    print("--- Starting Training ---")
    # This handles data loading, model init, training loop, and saving best model
    train_model()

    # 3. Validation & Failure Analysis
    print("--- Starting Validation & Failure Analysis ---")

    # Load the best model saved during training
    device = Config.DEVICE
    model = get_model()

    if os.path.exists(Config.CHECKPOINT_PATH):
        print(f"Loading weights from {Config.CHECKPOINT_PATH}")
        model.load_state_dict(torch.load(Config.CHECKPOINT_PATH, map_location=device))
    else:
        print(
            "Warning: No checkpoint found. Using random weights (Analysis will be meaningless)."
        )

    model.to(device)
    model.eval()

    # Get validation loader
    _, val_loader, _, _ = get_dataloaders(load_cached_data=True)

    total_val_score = 0.0
    num_batches = 0

    sample_errors = []
    sample_intensities = []

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Inference
            outputs = model(images)
            logits = outputs.logits

            # Upsample to match mask size
            logits = F.interpolate(
                logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )
            probs = torch.sigmoid(logits)

            # 1. Global Metric Calculation (Average of batch scores to match training logic)
            batch_score = fbeta_score(probs, masks, threshold=0.5, beta=0.5)
            total_val_score += batch_score
            num_batches += 1

            # 2. Data Collection for Failure Analysis
            # Iterate over samples in the batch
            for i in range(images.shape[0]):
                img = images[i]  # (C, H, W)
                msk = masks[i]  # (1, H, W)
                prb = probs[i]  # (1, H, W)

                # Calculate individual score
                # Note: fbeta_score flattens input, so passing single sample works
                s_score = fbeta_score(prb, msk, threshold=0.5, beta=0.5)

                # Error magnitude
                error = 1.0 - s_score

                # Input feature: Mean intensity
                # Images are normalized inputs
                intensity = img.mean().item()

                sample_errors.append(error)
                sample_intensities.append(intensity)

    # Compute Final Metric
    final_metric = total_val_score / num_batches if num_batches > 0 else 0.0

    # Print required metric
    print(f"Final Validation Metric: {final_metric}")

    # Compute Correlation
    if len(sample_errors) > 1:
        # Calculate Pearson correlation using numpy
        correlation = np.corrcoef(sample_errors, sample_intensities)[0, 1]
        print(
            f"Failure Analysis: Correlation between Error and Input Intensity: {correlation}"
        )
    else:
        print("Failure Analysis: Insufficient data for correlation.")

    # 4. Submission
    # Check against the specific threshold from the prompt
    THRESHOLD = 0.5511069832462687

    if final_metric > THRESHOLD:
        print(
            f"Metric {final_metric} exceeds threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission(
            test_metadata_path=Config.TEST_METADATA_PATH,
            checkpoint_path=Config.CHECKPOINT_PATH,
            output_path=Config.SUBMISSION_PATH,
        )
    else:
        print(
            f"Metric {final_metric} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
