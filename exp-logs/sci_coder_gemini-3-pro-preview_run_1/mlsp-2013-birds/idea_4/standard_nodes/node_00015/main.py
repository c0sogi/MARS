import sys
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.dataset import get_datasets
from library.model import BirdClassifier
from library.train import train_one_epoch, validate, inference
from library.utils import seed_everything, save_state, calculate_metric


def main():
    # ==========================================
    # 1. Setup & Initialization
    # ==========================================
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # ==========================================
    # 2. Data Loading
    # ==========================================
    # Uses cached data if available (load_cached_data=True)
    train_dataset, val_dataset, test_dataset = get_datasets(load_cached_data=True)

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

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    model = BirdClassifier(backbone=Config.BACKBONE, pretrained=Config.PRETRAINED)
    model = model.to(device)

    # Loss, Optimizer, Scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # ==========================================
    # 4. Training Loop
    # ==========================================
    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, Config.MIXUP_ALPHA
        )

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            save_state(model, best_model_path)

    # ==========================================
    # 5. Final Evaluation & Metric
    # ==========================================
    # Load best model state
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    val_targets = []
    val_preds = []
    val_img_stats = []  # For failure analysis

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)

            val_targets.append(labels.cpu().numpy())
            val_preds.append(probs.cpu().numpy())

            # Compute simple stats for failure analysis: Mean intensity and Std Dev
            # Images are (B, C, H, W). We aggregate over C, H, W.
            imgs_np = images.cpu().numpy()
            means = imgs_np.mean(axis=(1, 2, 3))
            stds = imgs_np.std(axis=(1, 2, 3))
            val_img_stats.append(np.stack([means, stds], axis=1))

    val_targets = np.concatenate(val_targets, axis=0)
    val_preds = np.concatenate(val_preds, axis=0)
    val_img_stats = np.concatenate(val_img_stats, axis=0)

    # Compute Final Metric
    final_metric = calculate_metric(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 6. Failure Analysis
    # ==========================================
    # Calculate Mean Absolute Error per sample (averaged across classes)
    # MAE is a proxy for how "wrong" the model was overall for that sample
    sample_errors = np.abs(val_targets - val_preds).mean(axis=1)

    img_means = val_img_stats[:, 0]
    img_stds = val_img_stats[:, 1]

    # Calculate correlations
    # Check for zero variance to avoid warnings
    if np.std(sample_errors) > 1e-9 and np.std(img_means) > 1e-9:
        corr_mean, _ = pearsonr(sample_errors, img_means)
    else:
        corr_mean = 0.0

    if np.std(sample_errors) > 1e-9 and np.std(img_stds) > 1e-9:
        corr_std, _ = pearsonr(sample_errors, img_stds)
    else:
        corr_std = 0.0

    print("Failure Analysis:")
    print(f"Correlation (Error vs Image Mean Intensity): {corr_mean}")
    print(f"Correlation (Error vs Image Contrast/Std): {corr_std}")

    # ==========================================
    # 7. Submission
    # ==========================================
    THRESHOLD = 0.9255537489325414

    if final_metric > THRESHOLD:
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        # inference function handles loading the best model and saving CSV
        inference(best_model_path, test_loader, device)
    else:
        print(
            f"Metric {final_metric} did not pass threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
