import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, update_bn
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
    swa_model = AveragedModel(model).to(Config.DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 4. Training Loop (SWA)
    print("Starting Training...")

    # Burn-in
    for epoch in range(1, Config.BURN_IN_EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, Config.DEVICE)
        # Silent progress as requested, or minimal print
        # print(f"Burn-in Epoch {epoch}: Loss {loss:.4f}")

    # SWA Phase
    for epoch in range(1, Config.SWA_EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, Config.DEVICE)
        swa_model.update_parameters(model)
        # print(f"SWA Epoch {epoch}: Loss {loss:.4f}")

    # Update BN
    print("Updating BN statistics...")
    update_bn(train_loader, swa_model, device=Config.DEVICE)

    # 5. Validation
    print("Validating...")
    swa_model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(Config.DEVICE)
            outputs = swa_model(images)
            probs = torch.softmax(outputs, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Metric (Mean column-wise ROC AUC)
    # targets are one-hot/probabilities. roc_auc_score handles this with multi_class='ovr'
    try:
        val_auc = roc_auc_score(
            all_targets, all_preds, average="macro", multi_class="ovr"
        )
    except ValueError:
        # Fallback if a class is missing in validation batch (unlikely with stratified split)
        val_auc = 0.5

    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    analyze_failures(swa_model, val_loader, Config.DEVICE, criterion)

    # 7. Submission
    THRESHOLD = 0.9871488489626378
    if val_auc > THRESHOLD:
        print("Validation metric meets threshold. Generating submission...")
        # We can reuse the predict_and_submit from engine, but we need to ensure
        # it uses the SWA model we just trained.
        predict_and_submit(swa_model)
    else:
        print(
            f"Validation metric {val_auc} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
