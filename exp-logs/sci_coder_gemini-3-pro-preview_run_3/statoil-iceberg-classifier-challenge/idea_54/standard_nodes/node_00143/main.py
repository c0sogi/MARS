import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss

# Import from provided library files
from library.utils import set_seed, get_device, save_checkpoint
from library.dataset import IcebergDataset
from library.model import SAICNN
from library.train import train_one_epoch, evaluate, predict


def analyze_failures(model, val_loader, device):
    """
    Performs failure analysis by correlating error magnitude with input features.
    """
    model.eval()
    all_targets = []
    all_probs = []
    all_angles = []
    all_b1_means = []
    all_b2_means = []

    with torch.no_grad():
        for (images, angles), targets in val_loader:
            images = images.to(device)
            angles_gpu = angles.to(device)

            # Forward pass
            outputs = model(images, angles_gpu)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            # Store data
            all_targets.extend(targets.numpy())
            all_probs.extend(probs)
            all_angles.extend(angles.numpy())

            # Compute image stats for analysis (Band 1 and Band 2 means)
            # images shape: (B, 3, 75, 75) -> index 0 is HH, 1 is HV
            b1_mean = torch.mean(images[:, 0, :, :], dim=(1, 2)).cpu().numpy()
            b2_mean = torch.mean(images[:, 1, :, :], dim=(1, 2)).cpu().numpy()

            all_b1_means.extend(b1_mean)
            all_b2_means.extend(b2_mean)

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    all_angles = np.array(all_angles)
    all_b1_means = np.array(all_b1_means)
    all_b2_means = np.array(all_b2_means)

    # Calculate Error Magnitude
    errors = np.abs(all_targets - all_probs)

    print("\n--- Failure Analysis ---")
    print(f"Average Absolute Error: {np.mean(errors):.6f}")

    # Correlations
    # Handle NaNs in angles if any remain (though dataset handles imputation, checking just in case)
    valid_mask = ~np.isnan(all_angles)
    if np.sum(valid_mask) > 0:
        corr_angle = np.corrcoef(errors[valid_mask], all_angles[valid_mask])[0, 1]
        print(f"Correlation (Error vs Inc Angle): {corr_angle:.6f}")

    corr_b1 = np.corrcoef(errors, all_b1_means)[0, 1]
    print(f"Correlation (Error vs Band 1 Mean): {corr_b1:.6f}")

    corr_b2 = np.corrcoef(errors, all_b2_means)[0, 1]
    print(f"Correlation (Error vs Band 2 Mean): {corr_b2:.6f}")
    print("------------------------\n")


def main():
    # 1. Setup
    SEED = 42
    set_seed(SEED)
    device = get_device()

    WORKING_DIR = "./working"
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Hyperparameters for Fast Baseline
    BATCH_SIZE = 32
    EPOCHS = 30  # Sufficient for convergence on this small dataset
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    THRESHOLD = 0.17174082291273365

    print(f"Using device: {device}")

    # 2. Data Loading
    # Using the provided metadata splits
    train_csv = "./metadata/train.csv"
    val_csv = "./metadata/val.csv"

    train_dataset = IcebergDataset(train_csv, mode="train", load_cached_data=True)
    val_dataset = IcebergDataset(val_csv, mode="val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = SAICNN().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_val_loss = float("inf")
    best_model_path = os.path.join(CHECKPOINT_DIR, "model_best_baseline.pth")

    print("Starting training...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        # Optional: Print progress (kept minimal)
        # print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    print("Training complete.")

    # 5. Validation Assessment
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Generate predictions on validation set for metric calculation
    val_probs = []
    val_targets = []

    with torch.no_grad():
        for (images, angles), targets in val_loader:
            images = images.to(device)
            angles = angles.to(device)

            outputs = model(images, angles)
            probs = torch.sigmoid(outputs)

            val_probs.extend(probs.cpu().numpy().flatten())
            val_targets.extend(targets.numpy())

    # Calculate Metric
    final_metric = log_loss(val_targets, val_probs)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    analyze_failures(model, val_loader, device)

    # 7. Submission Generation
    if final_metric < THRESHOLD:
        print(
            f"Metric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        test_csv = "./metadata/test.csv"
        test_dataset = IcebergDataset(test_csv, mode="test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        ids, preds = predict(model, test_loader, device)

        df_sub = pd.DataFrame({"id": ids, "is_iceberg": preds})
        sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        df_sub.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        print(
            f"Metric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
