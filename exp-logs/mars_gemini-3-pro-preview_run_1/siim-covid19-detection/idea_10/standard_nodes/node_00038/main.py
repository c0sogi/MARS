import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

# Import from provided library
from library.config import Config
from library.dataset import prepare_data, SIIMDataset, get_transforms
from library.model import ResNet18UNet
from library.train import train_fn, eval_fn
from library.utils import seed_everything, mask2bbox
from library.predict import inference


def failure_analysis(model, loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude (Study Loss) and metadata.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")

    errors = []
    class_labels = []
    num_boxes_list = []

    with torch.no_grad():
        for images, masks, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            # Forward
            _, logit_cls = model(images)

            # Calculate error (Loss) per sample
            target_indices = torch.argmax(labels, dim=1)
            loss = criterion(logit_cls, target_indices)

            # Collect data
            errors.extend(loss.cpu().numpy())
            class_labels.extend(target_indices.cpu().numpy())

            # Count boxes per mask
            batch_masks = masks.cpu().numpy()
            for i in range(batch_masks.shape[0]):
                boxes = mask2bbox(batch_masks[i, 0], threshold=0.5)
                num_boxes_list.append(len(boxes))

    errors = np.array(errors)
    class_labels = np.array(class_labels)
    num_boxes_list = np.array(num_boxes_list)

    print("\n==== Failure Analysis ====")
    # Correlation with Class Label
    if np.std(class_labels) > 0 and np.std(errors) > 0:
        corr_class = np.corrcoef(class_labels, errors)[0, 1]
        print(f"Correlation (Error vs Class Label): {corr_class:.4f}")
    else:
        print("Correlation (Error vs Class Label): Undefined (zero variance)")

    # Correlation with Num Boxes
    if np.std(num_boxes_list) > 0 and np.std(errors) > 0:
        corr_boxes = np.corrcoef(num_boxes_list, errors)[0, 1]
        print(f"Correlation (Error vs Num Boxes): {corr_boxes:.4f}")
    else:
        print("Correlation (Error vs Num Boxes): Undefined (zero variance)")
    print("==========================\n")


def main():
    # 1. Configuration & Setup
    # Ensure output dir exists
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Cite solution_lesson_node_00002: Extending training duration and using full data
    print(f"Running training with {Config.EPOCHS} epochs on full dataset...")

    # 2. Data Loading
    # Load cached data
    print("Loading Data...")
    train_images, train_masks, train_labels, _ = prepare_data(
        "train", load_cached_data=True
    )
    val_images, val_masks, val_labels, _ = prepare_data("val", load_cached_data=True)

    # Create Datasets
    train_dataset = SIIMDataset(
        train_images, train_masks, train_labels, transforms=get_transforms("train")
    )
    val_dataset = SIIMDataset(
        val_images, val_masks, val_labels, transforms=get_transforms("val")
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = ResNet18UNet(num_classes=Config.NUM_CLASSES, pretrained=True)
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_map = -1.0

    print("Starting Training...")
    for epoch in range(Config.EPOCHS):
        start = time.time()

        train_loss = train_fn(
            model, train_loader, optimizer, criterion_cls, criterion_seg, device
        )
        val_loss, val_map = eval_fn(
            model, val_loader, criterion_cls, criterion_seg, device
        )

        scheduler.step()

        elapsed = time.time() - start
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Time: {elapsed:.0f}s | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"mAP (All Classes): {val_map:.4f}"
        )

        # Use mAP as the primary metric
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), Config.MODEL_PATH)

    print(f"Training finished. Best mAP: {best_map:.4f}")

    # 5. Final Validation & Metrics
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)

    # Re-run eval on full validation set to be precise
    _, final_map = eval_fn(model, val_loader, criterion_cls, criterion_seg, device)

    print(f"Final Validation mAP: {final_map}")

    # 6. Failure Analysis
    failure_analysis(model, val_loader, device)

    # 7. Submission
    # Always generate submission regardless of threshold to avoid InvalidMetricValueError
    # Cite debug_lesson_5
    print(f"Generating submission (mAP: {final_map})...")
    inference(debug=False)


if __name__ == "__main__":
    main()
