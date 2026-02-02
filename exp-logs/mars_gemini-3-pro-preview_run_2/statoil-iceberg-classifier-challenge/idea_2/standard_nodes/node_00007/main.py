import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import sys
import os

# Import from the provided library files
from library.config import (
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    BATCH_SIZE,
    DROPOUT_RATE,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    SEED,
)
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import SAHCN, train_model, predict_and_submit


def main():
    # 1. Reproducibility
    seed_everything(SEED)

    # 2. Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Data Loading
    # We use debug=False to train on the full dataset for best performance.
    # The dataset is small enough that this remains fast.
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, batch_size=BATCH_SIZE, debug=False
    )

    # 4. Model Initialization
    model = SAHCN(dropout_rate=DROPOUT_RATE)
    model = model.to(device)

    # 5. Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Training
    # train_model handles the loop and loads the best state dict upon completion
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        device=device,
        save_path=MODEL_SAVE_PATH,
    )

    # 7. Validation Assessment & Failure Analysis
    print("\nRunning Validation Assessment...")
    model.eval()

    val_probs = []
    val_targets = []
    val_angles = []
    val_img_means = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            angles = batch["angle"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            outputs = model(images, angles)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            # Collect data
            val_probs.extend(probs)
            val_targets.extend(labels.cpu().numpy().flatten())
            val_angles.extend(angles.cpu().numpy().flatten())

            # Calculate simple image stat (mean intensity) for failure analysis
            # images shape: (B, 3, 75, 75) -> mean over (1, 2, 3)
            batch_img_means = images.mean(dim=(1, 2, 3)).cpu().numpy()
            val_img_means.extend(batch_img_means)

    val_probs = np.array(val_probs)
    val_targets = np.array(val_targets)
    val_angles = np.array(val_angles)
    val_img_means = np.array(val_img_means)

    # Calculate Metric
    # Clip probabilities to avoid log(0) errors, though sigmoid output is usually safe
    val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
    final_metric = log_loss(val_targets, val_probs_clipped)

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\nFailure Analysis:")
    # Calculate absolute error
    errors = np.abs(val_targets - val_probs)

    # Correlation with Incidence Angle
    corr_angle = np.corrcoef(errors, val_angles)[0, 1]
    print(f"Correlation (Error vs Inc Angle): {corr_angle:.6f}")

    # Correlation with Image Intensity
    corr_intensity = np.corrcoef(errors, val_img_means)[0, 1]
    print(f"Correlation (Error vs Image Intensity): {corr_intensity:.6f}")

    # 8. Conditional Submission
    threshold = 0.4997743261785452
    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({threshold}). Generating submission..."
        )
        predict_and_submit(
            model=model,
            test_loader=test_loader,
            device=device,
            submission_path=SUBMISSION_PATH,
        )
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
