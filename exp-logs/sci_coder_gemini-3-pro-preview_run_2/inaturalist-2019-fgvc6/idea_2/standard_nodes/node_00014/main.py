import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, Subset
from PIL import Image
from scipy.stats import pointbiserialr

from library.config import Config
from library.dataset import INatDataset, get_transforms
from library.model import create_model
from library.engine import train_one_epoch, validate, inference
from library.utils import set_seed, save_checkpoint


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Initializing Datasets...")
    # Training Data
    # Using full dataset as per Lesson 00004 (Maximize Global Accuracy via Full Distribution Training)
    # and Lesson 00012 (Prioritize Data Volume over Model Scale).
    train_dataset = INatDataset(
        Config.TRAIN_METADATA, transform=get_transforms("train", Config.IMAGE_SIZE)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Validation Data
    val_dataset = INatDataset(
        Config.VAL_METADATA, transform=get_transforms("val", Config.IMAGE_SIZE)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print(f"Creating {Config.MODEL_NAME} model...")
    model = create_model(Config.NUM_CLASSES, Config.PRETRAINED)
    model.to(device)

    # 4. Training Setup
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Run for Config.NUM_EPOCHS (18) to ensure convergence (Lesson 00005)
    num_epochs = Config.NUM_EPOCHS
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=Config.ETA_MIN
    )

    scaler = torch.cuda.amp.GradScaler()

    # 5. Training Loop
    best_acc = 0.0
    print("Starting training...")

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            train_loader, model, criterion, optimizer, scaler, device, epoch
        )
        val_loss, val_acc = validate(val_loader, model, criterion, device)

        scheduler.step()

        # Save Checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_acc": best_acc,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
        )

    # 6. Final Evaluation & Metric
    print("Loading best model for final evaluation...")
    checkpoint = torch.load(Config.BEST_MODEL_PATH)
    model.load_state_dict(checkpoint["state_dict"])

    val_loss, val_acc = validate(val_loader, model, criterion, device)

    # Metric is Top-1 Error (fraction)
    # val_acc is percentage (0-100), so error % is 100 - val_acc
    # error fraction is (100 - val_acc) / 100.0
    final_metric_fraction = (100.0 - val_acc) / 100.0

    print(f"Final Validation Metric: {final_metric_fraction}")

    # 7. Failure Analysis
    print("Performing Failure Analysis...")
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    # Error vector: 1 if incorrect, 0 if correct
    errors = (all_preds != all_targets).astype(int)

    # Extract image features (Width, Height, AR)
    # We read headers from disk.
    val_df = pd.read_csv(Config.VAL_METADATA)
    widths = []
    heights = []

    # To save time, we can process a subset if val set is huge,
    # but 46k is manageable (~2-3 mins).
    for file_name in val_df["file_name"]:
        path = os.path.join(Config.INPUT_DIR, file_name)
        try:
            with Image.open(path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except Exception:
            widths.append(0)
            heights.append(0)

    widths = np.array(widths)
    heights = np.array(heights)

    # Avoid division by zero
    mask = heights > 0
    aspect_ratios = np.zeros_like(widths, dtype=float)
    aspect_ratios[mask] = widths[mask] / heights[mask]

    # Compute Point-Biserial Correlation
    # We only compute for valid images
    valid_indices = mask

    if np.sum(valid_indices) > 0:
        corr_w = pointbiserialr(errors[valid_indices], widths[valid_indices])
        corr_h = pointbiserialr(errors[valid_indices], heights[valid_indices])
        corr_ar = pointbiserialr(errors[valid_indices], aspect_ratios[valid_indices])

        print(
            f"Correlation Error vs Width: {corr_w.correlation:.4f} (p={corr_w.pvalue:.4f})"
        )
        print(
            f"Correlation Error vs Height: {corr_h.correlation:.4f} (p={corr_h.pvalue:.4f})"
        )
        print(
            f"Correlation Error vs Aspect Ratio: {corr_ar.correlation:.4f} (p={corr_ar.pvalue:.4f})"
        )
    else:
        print("Could not compute correlations due to missing image data.")

    # 8. Submission
    # Threshold check: 0.2647424892703862
    threshold = 0.2647424892703862

    if final_metric_fraction < threshold:
        print(
            f"Validation Error ({final_metric_fraction:.4f}) < Threshold ({threshold:.4f}). Generating submission..."
        )

        test_dataset = INatDataset(
            Config.TEST_METADATA,
            transform=get_transforms("test", Config.IMAGE_SIZE),
            mode="test",
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        inference(test_loader, model, device)
    else:
        print(
            f"Validation Error ({final_metric_fraction:.4f}) >= Threshold ({threshold:.4f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
