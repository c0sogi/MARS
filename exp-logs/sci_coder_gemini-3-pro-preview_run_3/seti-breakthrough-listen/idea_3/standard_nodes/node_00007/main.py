import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import from provided library
from library.config import Config
from library.dataset import get_dataloaders
from library.model import Shallow3DCNN
from library.engine import train_one_epoch, valid_one_epoch, predict_with_tta
from library.utils import set_seed, get_score


def main():
    # --- 1. Setup ---
    warnings.filterwarnings("ignore")
    Config.setup()
    set_seed(Config.SEED)

    # Hyperparameters for fast baseline
    EPOCHS = 5

    # --- 2. Data Loading ---
    # Using defaults from Config (Batch Size 64, etc.)
    train_loader, val_loader, test_loader = get_dataloaders(
        train_batch_size=Config.BATCH_SIZE,
        val_batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # --- 3. Model Initialization ---
    device = Config.DEVICE
    model = Shallow3DCNN().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    criterion = nn.BCEWithLogitsLoss()

    # --- 4. Training Loop ---
    best_auc = 0.0

    for epoch in range(EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validation
        val_loss, val_auc = valid_one_epoch(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_PATH)

    # --- 5. Final Validation & Failure Analysis ---
    # Load best model
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    model.eval()
    val_preds = []
    val_targets = []

    # Accumulators for failure analysis stats
    stats = {
        "mean_intensity": [],
        "std_intensity": [],
        "max_intensity": [],
        "on_off_diff": [],
        "error": [],
    }

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(images)
            probs = torch.sigmoid(outputs).view(-1)

            # To CPU
            preds_np = probs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            images_np = images.cpu().numpy()  # Shape: (B, 1, 6, 256, 256)

            # Store predictions
            val_preds.extend(preds_np.tolist())
            val_targets.extend(targets_np.tolist())

            # Calculate errors
            batch_errors = np.abs(targets_np - preds_np)

            # Extract features for failure analysis
            # Iterate over batch
            for i in range(images_np.shape[0]):
                # Get the 6-depth image: shape (6, 256, 256)
                img = images_np[i, 0]

                # Features
                stats["mean_intensity"].append(np.mean(img))
                stats["std_intensity"].append(np.std(img))
                stats["max_intensity"].append(np.max(img))

                # On-Target (0, 2, 4) vs Off-Target (1, 3, 5)
                on_target_mean = np.mean(img[[0, 2, 4]])
                off_target_mean = np.mean(img[[1, 3, 5]])
                stats["on_off_diff"].append(on_target_mean - off_target_mean)

                stats["error"].append(batch_errors[i])

    # Compute Final Metric
    final_auc = get_score(np.array(val_targets), np.array(val_preds))
    print(f"Final Validation Metric: {final_auc}")

    # Compute Correlations
    df_stats = pd.DataFrame(stats)
    correlations = df_stats.corr()["error"].drop("error")

    print("Failure Analysis (Correlation with Error Magnitude):")
    print(correlations)

    # --- 6. Submission ---
    THRESHOLD = 0.5196359687502365

    if final_auc > THRESHOLD:
        # Generate predictions on test set
        test_preds = predict_with_tta(model, test_loader, device)

        # Load test metadata to ensure ID alignment
        df_sub = pd.read_csv(Config.TEST_METADATA)
        df_sub["target"] = test_preds

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        df_sub[["id", "target"]].to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        pass


if __name__ == "__main__":
    main()
