import os
import sys
import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.dataset import BirdDataset, get_transforms
from library.model import get_model
from library.trainer import train_one_epoch, validate, predict
from library.utils import set_seed, calculate_multilabel_auc, save_submission


def main():
    # 1. Configuration and Setup
    # Limit epochs for a fast baseline execution as requested
    Config.EPOCHS = 20

    set_seed(Config.SEED)
    device = Config.DEVICE

    # 2. Data Loading
    train_transform = get_transforms(mode="train")
    val_transform = get_transforms(mode="val")

    # Load datasets with caching enabled
    train_dataset = BirdDataset(
        metadata_path=Config.TRAIN_CSV,
        mode="train",
        transform=train_transform,
        load_cached_data=True,
    )
    val_dataset = BirdDataset(
        metadata_path=Config.VAL_CSV,
        mode="val",
        transform=val_transform,
        load_cached_data=True,
    )
    test_dataset = BirdDataset(
        metadata_path=Config.TEST_CSV,
        mode="test",
        transform=val_transform,
        load_cached_data=True,
    )

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
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = get_model(
        model_name=Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT_RATE,
    )
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # 4. Training Loop
    best_auc = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = Config.PATIENCE
    epochs_no_improve = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            scheduler,
            mixup_alpha=0.4,
        )

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}"
        )

        # Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            # Save best model to disk
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    # 5. Final Validation & Metric Reporting
    print("Loading best model for final evaluation...")
    model.load_state_dict(best_model_wts)
    model.eval()

    # We need detailed predictions for failure analysis, so we iterate manually
    all_preds = []
    all_targets = []
    all_image_means = []
    all_label_counts = []

    # Use val_loader to get data
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

            # Collect features for failure analysis
            # Feature 1: Image Mean Intensity (Signal strength)
            # images is (B, 3, H, W). We take mean over dimensions (1, 2, 3)
            img_means = torch.mean(images, dim=(1, 2, 3))
            all_image_means.append(img_means.cpu().numpy())

            # Feature 2: Label Count (Complexity)
            lbl_counts = torch.sum(labels, dim=1)
            all_label_counts.append(lbl_counts.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    all_image_means = np.concatenate(all_image_means)
    all_label_counts = np.concatenate(all_label_counts)

    final_metric = calculate_multilabel_auc(all_targets, all_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate Error Magnitude (Mean Absolute Error per sample)
    # Shape: (N_samples, N_classes) -> mean over classes -> (N_samples,)
    errors = np.abs(all_preds - all_targets)
    mean_errors = np.mean(errors, axis=1)

    # Correlation with Signal Intensity
    corr_signal = np.corrcoef(mean_errors, all_image_means)[0, 1]
    print(f"Correlation between Error and Signal Intensity: {corr_signal:.4f}")

    # Correlation with Label Complexity
    corr_complexity = np.corrcoef(mean_errors, all_label_counts)[0, 1]
    print(f"Correlation between Error and Label Count: {corr_complexity:.4f}")

    # 7. Conditional Submission
    THRESHOLD = 0.6514627421758025

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate predictions on test set
        predictions, test_rec_ids = predict(model, test_loader, device)

        # Save submission
        save_submission(predictions, test_rec_ids, Config.PREDICTIONS_PATH)
        print(f"Submission saved to {Config.PREDICTIONS_PATH}")
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
