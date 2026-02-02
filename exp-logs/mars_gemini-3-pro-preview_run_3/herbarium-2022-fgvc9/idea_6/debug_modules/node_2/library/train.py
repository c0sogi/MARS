import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from library.config import Config
from library.dataset import get_dataloader
from library.model import HierarchicalEfficientNet
from library.utils import set_seed


def train_one_epoch(model, loader, optimizer, scheduler, device, criterion, epoch):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: The training DataLoader.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        device: The computing device (CPU/GPU).
        criterion: The loss function (CrossEntropyLoss).
        epoch: Current epoch number.

    Returns:
        avg_loss: The average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)

        # Move targets to device
        labels_species = targets["species"].to(device)
        labels_genus = targets["genus"].to(device)
        labels_family = targets["family"].to(device)

        batch_size = images.size(0)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate losses for each head
        loss_species = criterion(outputs["species"], labels_species)
        loss_genus = criterion(outputs["genus"], labels_genus)
        loss_family = criterion(outputs["family"], labels_family)

        # Weighted sum of losses
        total_loss = (
            Config.WEIGHT_SPECIES * loss_species
            + Config.WEIGHT_GENUS * loss_genus
            + Config.WEIGHT_FAMILY * loss_family
        )

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += total_loss.item() * batch_size
        dataset_size += batch_size

    avg_loss = running_loss / dataset_size
    return avg_loss


def validate(model, loader, device, criterion):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: The validation DataLoader.
        device: The computing device.
        criterion: The loss function.

    Returns:
        avg_loss: Average validation loss.
        f1_macro: Macro F1 score for the Species head.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)

            labels_species = targets["species"].to(device)
            labels_genus = targets["genus"].to(device)
            labels_family = targets["family"].to(device)

            batch_size = images.size(0)

            outputs = model(images)

            loss_species = criterion(outputs["species"], labels_species)
            loss_genus = criterion(outputs["genus"], labels_genus)
            loss_family = criterion(outputs["family"], labels_family)

            total_loss = (
                Config.WEIGHT_SPECIES * loss_species
                + Config.WEIGHT_GENUS * loss_genus
                + Config.WEIGHT_FAMILY * loss_family
            )

            running_loss += total_loss.item() * batch_size
            dataset_size += batch_size

            # Collect predictions for Species F1 Score
            preds = torch.argmax(outputs["species"], dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels_species.cpu().numpy())

    avg_loss = running_loss / dataset_size

    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate Macro F1 Score
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, f1_macro


def run_training():
    """
    Orchestrates the two-stage training pipeline.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 1. Load Metadata
    print("Loading metadata...")
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    # 2. Initialize Model and Loss
    print(f"Initializing model: {Config.MODEL_NAME}")
    model = HierarchicalEfficientNet(pretrained=Config.PRETRAINED)
    model = model.to(device)

    # Shared loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    # -------------------------------------------------------------------------
    # STAGE 1: Feature Learning (224x224)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print(f"STAGE 1: Training at {Config.STAGE_1_RES}x{Config.STAGE_1_RES}")
    print("=" * 40)

    train_loader_s1 = get_dataloader(
        df_train,
        mode="train",
        batch_size=Config.STAGE_1_BATCH_SIZE,
        image_size=Config.STAGE_1_RES,
        shuffle=True,
    )
    val_loader_s1 = get_dataloader(
        df_val,
        mode="valid",
        batch_size=Config.STAGE_1_BATCH_SIZE,
        image_size=Config.STAGE_1_RES,
        shuffle=False,
    )

    optimizer_s1 = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler_s1 = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_s1,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader_s1),
        epochs=Config.STAGE_1_EPOCHS,
        pct_start=0.1,
    )

    best_f1_s1 = 0.0

    for epoch in range(Config.STAGE_1_EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader_s1, optimizer_s1, scheduler_s1, device, criterion, epoch
        )
        val_loss, val_f1 = validate(model, val_loader_s1, device, criterion)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.STAGE_1_EPOCHS} - "
            f"Time: {elapsed:.0f}s - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - "
            f"Val F1 (Macro): {val_f1:.10f}"
        )

        if val_f1 > best_f1_s1:
            print(f"New best Stage 1 F1: {val_f1:.10f}. Saving checkpoint...")
            best_f1_s1 = val_f1
            torch.save(model.state_dict(), Config.CHECKPOINT_STAGE_1)

    print(f"Stage 1 completed. Best F1: {best_f1_s1:.10f}")

    # -------------------------------------------------------------------------
    # STAGE 2: Fine-Grained Refinement (320x320)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print(f"STAGE 2: Fine-tuning at {Config.STAGE_2_RES}x{Config.STAGE_2_RES}")
    print("=" * 40)

    # Load best weights from Stage 1
    print("Loading best weights from Stage 1...")
    model.load_state_dict(torch.load(Config.CHECKPOINT_STAGE_1, map_location=device))

    train_loader_s2 = get_dataloader(
        df_train,
        mode="train",
        batch_size=Config.STAGE_2_BATCH_SIZE,
        image_size=Config.STAGE_2_RES,
        shuffle=True,
    )
    val_loader_s2 = get_dataloader(
        df_val,
        mode="valid",
        batch_size=Config.STAGE_2_BATCH_SIZE,
        image_size=Config.STAGE_2_RES,
        shuffle=False,
    )

    # Re-initialize optimizer and scheduler for Stage 2
    # We use the same max_lr but the OneCycle schedule will adapt to the new steps/epochs
    optimizer_s2 = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler_s2 = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_s2,
        max_lr=Config.LEARNING_RATE,  # Could potentially lower this, but OneCycle handles warmup/decay
        steps_per_epoch=len(train_loader_s2),
        epochs=Config.STAGE_2_EPOCHS,
        pct_start=0.1,
    )

    best_f1_s2 = 0.0  # Track best in this stage

    for epoch in range(Config.STAGE_2_EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader_s2, optimizer_s2, scheduler_s2, device, criterion, epoch
        )
        val_loss, val_f1 = validate(model, val_loader_s2, device, criterion)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.STAGE_2_EPOCHS} - "
            f"Time: {elapsed:.0f}s - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - "
            f"Val F1 (Macro): {val_f1:.10f}"
        )

        if val_f1 > best_f1_s2:
            print(f"New best Stage 2 F1: {val_f1:.10f}. Saving checkpoint...")
            best_f1_s2 = val_f1
            torch.save(model.state_dict(), Config.CHECKPOINT_STAGE_2)

    print(f"Stage 2 completed. Best F1: {best_f1_s2:.10f}")
    print("Training pipeline finished.")
