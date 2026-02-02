import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import from provided libraries
import library.config as config
from library.utils import set_seed, AverageMeter
from library.data_processing import load_data
from library.dataset import SICAVDataset, get_transforms
from library.model import SICAVModel

# ==========================================
# Configuration Overrides for Fast Baseline
# ==========================================
# Limit epochs to ensure execution within time limits
config.EPOCHS = 5
# Ensure we use the working directory defined in config
WORKING_DIR = config.WORKING_DIR
BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_baseline_model.pth")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = AverageMeter()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

            losses.update(loss.item(), images.size(0))
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    return losses.avg, np.array(all_probs), np.array(all_labels)


def main():
    # 1. Setup
    set_seed(config.SEED)
    device = config.DEVICE
    print(f"Using device: {device}")

    # 2. Load Data
    # We load train and val separately as required by the task description
    print("Loading datasets...")
    train_ids, train_images, train_labels = load_data("train", load_cached_data=True)
    val_ids, val_images, val_labels = load_data("val", load_cached_data=True)

    # Limit training samples if needed for extreme speed (optional, using full here as it's small enough)
    # train_images = train_images[:500]
    # train_labels = train_labels[:500]

    # Create Datasets
    train_dataset = SICAVDataset(
        train_ids, train_images, train_labels, transforms=get_transforms("train")
    )
    val_dataset = SICAVDataset(
        val_ids, val_images, val_labels, transforms=get_transforms("val")
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Initialize Model
    print("Initializing SICAV Model...")
    model = SICAVModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # 4. Training Loop
    best_auc = 0.0
    print(f"Starting training for {config.EPOCHS} epochs...")

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_probs, val_targets = validate(
            model, val_loader, criterion, device
        )

        # Calculate AUC
        if len(np.unique(val_targets)) > 1:
            val_auc = roc_auc_score(val_targets, val_probs)
        else:
            val_auc = 0.5

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    # 5. Final Validation on Hold-out Set
    print("\nPerforming Final Validation...")
    # Load best model
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()

    _, final_probs, final_labels = validate(model, val_loader, criterion, device)

    if len(np.unique(final_labels)) > 1:
        final_metric = roc_auc_score(final_labels, final_probs)
    else:
        final_metric = 0.5

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate error magnitude
    errors = np.abs(final_labels - final_probs)

    # Extract simple features from validation images for correlation
    # We compute mean intensity and std dev of the input tensors
    # val_images is (N, H, W, C)
    val_means = np.mean(val_images, axis=(1, 2, 3))
    val_stds = np.std(val_images, axis=(1, 2, 3))

    # Calculate correlations
    corr_mean = np.corrcoef(errors, val_means)[0, 1]
    corr_std = np.corrcoef(errors, val_stds)[0, 1]

    print(f"Correlation between Error and Input Mean Intensity: {corr_mean:.4f}")
    print(f"Correlation between Error and Input Std Dev: {corr_std:.4f}")

    # 7. Submission
    THRESHOLD = 0.6705454545454544

    if final_metric > THRESHOLD:
        print(
            f"\nValidation metric {final_metric} > {THRESHOLD}. Generating submission..."
        )

        # Load Test Data
        test_ids, test_images, test_labels = load_data("test", load_cached_data=True)

        test_dataset = SICAVDataset(
            test_ids, test_images, test_labels, transforms=get_transforms("test")
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        # Inference
        all_test_probs = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits)
                all_test_probs.extend(probs.cpu().numpy().flatten())

        # Save Submission
        submission_df = pd.DataFrame(
            {"BraTS21ID": test_ids, "MGMT_value": all_test_probs}
        )

        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")
        print(submission_df.head())

    else:
        print(
            f"\nValidation metric {final_metric} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
