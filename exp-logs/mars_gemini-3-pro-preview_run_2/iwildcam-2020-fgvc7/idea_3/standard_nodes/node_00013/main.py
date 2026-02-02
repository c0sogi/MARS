import os
import sys
import torch
import pandas as pd
import numpy as np

# Import Config first to patch it for the fast baseline run
from library.config import Config

# ------------------------------------------------------------------------------
# Configuration Patching
# ------------------------------------------------------------------------------
# Override settings to ensure execution within 10 minutes
Config.EPOCHS = 2
Config.DEBUG = True
Config.DEBUG_SAMPLE_SIZE = 1000  # Small sample size for quick execution
Config.BATCH_SIZE = 16  # Reduced batch size for safety
Config.NUM_WORKERS = 2  # Reduced workers to lower overhead

# Initialize directories and seeds via Config
Config.setup()

from library.trainer import Trainer
from library.utils import seed_everything


def main():
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # --------------------------------------------------------------------------
    # 1. Training
    # --------------------------------------------------------------------------
    print("Initializing Trainer...")
    trainer = Trainer(debug=Config.DEBUG)

    print("Starting Training...")
    trainer.fit()

    # --------------------------------------------------------------------------
    # 2. Validation & Failure Analysis
    # --------------------------------------------------------------------------
    print("Running Validation and Failure Analysis...")

    # Set model to evaluation mode
    trainer.model.eval()
    val_loader = trainer.val_loader

    all_preds = []
    all_targets = []
    all_losses = []

    # Use reduction='none' to get individual sample losses
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)

            # Forward pass
            outputs = trainer.model(images)
            loss = criterion(outputs, targets)

            # Get predictions
            preds = torch.argmax(outputs, dim=1)

            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_losses.extend(loss.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_losses = np.array(all_losses)

    # Calculate and Print Metric
    accuracy = (all_preds == all_targets).mean()
    print(f"Final Validation Metric: {accuracy}")

    # Failure Analysis
    print("Performing Failure Analysis...")
    # Access the dataframe used in validation (subsampled if debug is True)
    val_df = val_loader.dataset.df.copy()

    if len(val_df) == len(all_losses):
        val_df["loss"] = all_losses
        val_df["correct"] = (all_preds == all_targets).astype(int)
        val_df["error"] = 1 - val_df["correct"]

        # Calculate correlation between Error and Input Features (Width, Height)
        correlations = val_df[["error", "width", "height"]].corr()["error"]

        print(
            f"Correlation between Error and Width: {correlations.get('width', 0):.4f}"
        )
        print(
            f"Correlation between Error and Height: {correlations.get('height', 0):.4f}"
        )
    else:
        print(
            "Warning: Mismatch between validation set size and predictions. Skipping detailed analysis."
        )

    # --------------------------------------------------------------------------
    # 3. Submission Generation
    # --------------------------------------------------------------------------
    THRESHOLD = 0.7304232880255179

    if accuracy > THRESHOLD:
        print(
            f"Validation metric ({accuracy:.16f}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(trainer.model, trainer.test_loader)
    else:
        print(
            f"Validation metric ({accuracy:.16f}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


def generate_submission(model, test_loader):
    """
    Generates submission file using Test Time Augmentation (TTA).
    TTA: Average of Original and Horizontal Flip.
    """
    model.eval()
    preds = []

    # Extract IDs from the dataset dataframe
    ids = test_loader.dataset.df["id"].values

    print("Running Inference on Test Set with TTA...")
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(Config.DEVICE)

            # TTA 1: Original
            logits1 = model(images)

            # TTA 2: Horizontal Flip (flip width dimension, index 3)
            images_flip = torch.flip(images, [3])
            logits2 = model(images_flip)

            # Average Logits
            avg_logits = (logits1 + logits2) / 2.0

            # Prediction
            batch_preds = torch.argmax(avg_logits, dim=1)
            preds.extend(batch_preds.cpu().numpy())

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"Id": ids, "Predicted": preds})

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


if __name__ == "__main__":
    main()
