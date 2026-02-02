import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import set_seed, get_device
from library.dataset import get_dataloaders
from library.model import EA_IDPH_CNN
from library.engine import train_one_epoch, evaluate


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = get_device()

    # 2. Data Loading
    # load_cached_data=True attempts to load pre-processed .npy files from cache
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = EA_IDPH_CNN().to(device)

    # 4. Optimizer and Loss
    # Using AdamW with constant learning rate and weight decay as per Idea 61
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "model_best.pth")

    for epoch in range(Config.EPOCHS):
        # Train one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Checkpointing and Early Stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            break

    # 6. Final Evaluation & Failure Analysis
    # Load the best model weights
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Collect all predictions, targets, and angles for detailed analysis
    all_probs = []
    all_targets = []
    all_angles = []

    with torch.no_grad():
        for images, angles, labels in val_loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            outputs = model(images, angles)
            probs = torch.sigmoid(outputs)

            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_angles.extend(angles.cpu().numpy())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    all_angles = np.array(all_angles).flatten()

    # Calculate Final Log Loss
    # Clip probabilities to avoid log(0) errors
    epsilon = 1e-15
    probs_clipped = np.clip(all_probs, epsilon, 1 - epsilon)
    final_log_loss = -np.mean(
        all_targets * np.log(probs_clipped)
        + (1 - all_targets) * np.log(1 - probs_clipped)
    )

    # Required Output
    print(f"Final Validation Metric: {final_log_loss}")

    # Failure Analysis: Correlation between error magnitude and incidence angle
    errors = np.abs(all_probs - all_targets)

    if np.std(all_angles) > 1e-6:
        correlation = np.corrcoef(errors, all_angles)[0, 1]
        print(
            f"Failure Analysis: Correlation between error and incidence angle: {correlation:.4f}"
        )
    else:
        print(
            "Failure Analysis: Incidence angles have insufficient variance for correlation analysis."
        )

    # 7. Submission Generation
    # Only generate submission if metric is below the specified threshold
    THRESHOLD = 0.17174082291273365

    if final_log_loss < THRESHOLD:
        test_ids = []
        test_probs = []

        with torch.no_grad():
            for images, angles, ids in test_loader:
                images = images.to(device)
                angles = angles.to(device)

                outputs = model(images, angles)
                probs = torch.sigmoid(outputs)

                test_probs.extend(probs.cpu().numpy())
                test_ids.extend(ids)

        # Create submission DataFrame
        submission_df = pd.DataFrame({"id": test_ids, "is_iceberg": test_probs})

        # Save to the specified submission path
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)


if __name__ == "__main__":
    main()
