import os
import sys
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything
from library.train import Trainer
from library.dataset import get_dataloaders


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Override Config for a fast baseline execution
    # Reducing epochs to 5 ensures we finish well within the 2-hour limit
    # while still allowing the model to converge reasonably well on the full dataset.
    Config.EPOCHS = 5

    # We use the full dataset (DEBUG=False) because the A100 is fast enough
    # to handle 120k images x 5 epochs in < 20 minutes.
    Config.DEBUG = False

    print(f"Configuration: Epochs={Config.EPOCHS}, Device={Config.DEVICE}")

    # ==========================================
    # 2. Training Pipeline
    # ==========================================
    # Initialize Trainer
    trainer = Trainer()

    # Execute training
    # This handles:
    # - Training loop
    # - Validation checkpointing
    # - Loading the best model at the end
    # - Threshold optimization
    # - Generating submission.csv
    trainer.fit()

    # ==========================================
    # 3. Validation Assessment
    # ==========================================
    print("\n--- Running Final Validation Assessment ---")

    # Get DataLoaders (using cached metadata for speed)
    _, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
        debug=Config.DEBUG,
    )

    # Run inference on validation set using the best threshold found during training
    # trainer.fit() ensures the best model is loaded in trainer.model
    val_loss, val_f1, val_probs, val_targets = trainer.validate(
        val_loader, threshold=trainer.best_threshold
    )

    # Print the required metric in full precision
    print(f"Final Validation Metric: {val_f1}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\n--- Running Failure Analysis ---")

    # We want to correlate error magnitude with input features (e.g., pixel stats).
    # We need to compute image statistics for the validation set.

    img_means = []
    img_stds = []

    # Iterate through validation loader to get image stats
    # We don't need gradients
    with torch.no_grad():
        for images, _ in val_loader:
            # images shape: (B, 3, H, W)
            # Flatten spatial dimensions to calculate stats per image
            B = images.size(0)
            flat_imgs = images.view(B, -1)  # (B, 3*H*W)

            # Calculate mean and std for each image in the batch
            batch_means = flat_imgs.mean(dim=1).cpu().numpy()
            batch_stds = flat_imgs.std(dim=1).cpu().numpy()

            img_means.extend(batch_means)
            img_stds.extend(batch_stds)

    img_means = np.array(img_means)
    img_stds = np.array(img_stds)

    # Calculate Error Magnitude per sample
    # Metric: Mean Absolute Error (MAE) between predicted probabilities and binary targets
    # This captures how far off the probability was from the truth (0 or 1).
    # Shape of val_probs and val_targets is (N_samples, N_classes)
    error_matrix = np.abs(val_targets - val_probs)
    sample_errors = np.mean(
        error_matrix, axis=1
    )  # Average error across all classes for each image

    # Create a DataFrame for correlation analysis
    analysis_df = pd.DataFrame(
        {"error": sample_errors, "pixel_mean": img_means, "pixel_std": img_stds}
    )

    # Calculate correlations
    correlations = analysis_df.corr()["error"].drop("error")

    print("Correlation between Model Error and Input Features:")
    print(correlations)

    # ==========================================
    # 5. Submission Verification
    # ==========================================
    # Trainer.fit() calls generate_submission at the end.
    # We verify the file exists.
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"\nSubmission successfully generated at: {Config.SUBMISSION_PATH}")
    else:
        print("\nWarning: Submission file not found. Regenerating...")
        trainer.generate_submission(test_loader)


if __name__ == "__main__":
    main()
