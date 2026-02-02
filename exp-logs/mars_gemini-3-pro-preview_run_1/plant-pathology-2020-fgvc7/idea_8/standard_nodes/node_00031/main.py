import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
import cv2

# Import from library
from library.config import Config
from library.utils import seed_everything, calculate_class_weights
from library.dataset import AppleDataset, get_transforms
from library.model import get_model
from library.engine import train_one_epoch, predict_and_submit


def analyze_failures(model, val_loader, device, criterion):
    """
    Performs failure analysis by correlating error magnitude with image intensity.
    """
    model.eval()
    losses = []
    intensities = []

    # We need to access the original images or compute stats.
    # Since the loader returns normalized tensors, we can compute intensity from the tensor
    # (approximation) or read from file. Using tensor is faster.

    # Un-normalization constants for approximate intensity reconstruction
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_indices = torch.argmax(targets, dim=1)

            # Forward pass
            outputs = model(images)

            # Calculate loss per sample (reduction='none')
            batch_losses = nn.CrossEntropyLoss(reduction="none")(
                outputs, target_indices
            )
            losses.extend(batch_losses.cpu().numpy())

            # Calculate mean intensity per image
            # Revert normalization: img * std + mean
            orig_imgs = images * std + mean
            # Mean over C, H, W
            batch_intensities = orig_imgs.mean(dim=(1, 2, 3))
            intensities.extend(batch_intensities.cpu().numpy())

    losses = np.array(losses)
    intensities = np.array(intensities)

    # Calculate correlation
    if len(losses) > 1:
        correlation = np.corrcoef(losses, intensities)[0, 1]
    else:
        correlation = 0.0

    print(
        f"Failure Analysis - Correlation between Error Magnitude and Image Intensity: {correlation:.4f}"
    )


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # Override Config to ensure we use the split for validation
    Config.USE_FULL_DATA = False

    print(f"Device: {Config.DEVICE}")

    # 2. Data Loading
    # Load metadata manually to ensure strict separation
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Create Datasets
    train_dataset = AppleDataset(
        df=train_df, transforms=get_transforms("train"), root_dir=Config.INPUT_DIR
    )
    val_dataset = AppleDataset(
        df=val_df, transforms=get_transforms("valid"), root_dir=Config.INPUT_DIR
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

    # Calculate Class Weights
    class_weights = calculate_class_weights(
        Config.TRAIN_METADATA_PATH,
        Config.CLASS_LABELS,
        load_cached_data=False,  # Force recalculate for the split
    )

    # 3. Model Initialization
    model = get_model(pretrained=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cite solution_lesson_node_00015: Synchronize scheduler cycle with total epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 4. Training Loop
    print("Starting Training...")

    for epoch in range(1, Config.EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, Config.DEVICE)
        scheduler.step()
        # print(f"Epoch {epoch}: Loss {loss:.4f}")

    # 5. Validation
    print("Validating...")
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(Config.DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Metric (Mean column-wise ROC AUC)
    try:
        val_auc = roc_auc_score(
            all_targets, all_preds, average="macro", multi_class="ovr"
        )
    except ValueError:
        val_auc = 0.5

    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    analyze_failures(model, val_loader, Config.DEVICE, criterion)

    # 7. Submission
    THRESHOLD = 0.9871488489626378
    if val_auc > THRESHOLD:
        print("Validation metric meets threshold. Generating submission...")
        predict_and_submit(model)
    else:
        print(
            f"Validation metric {val_auc} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
