import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
from library.config import Config
from library.utils import (
    seed_everything,
    calculate_macro_f1,
    generate_submission,
    compute_class_priors,
)
from library.dataset import get_dataloaders
from library.model import HierarchicalConvNeXt


def train_one_epoch(
    model, loader, optimizer, criterion_dict, device, scaler, scheduler=None
):
    """
    Trains the model for one epoch.
    Handles multi-task loss calculation and AMP scaling.
    """
    model.train()
    running_loss = 0.0

    # Iterate over batches
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)

        # Unpack labels tuple: (species, family, order)
        species_labels = labels[0].to(device)
        family_labels = labels[1].to(device)
        order_labels = labels[2].to(device)

        optimizer.zero_grad()

        # Mixed Precision Forward Pass
        with autocast():
            outputs = model(images)

            # Calculate loss for each head
            loss_species = criterion_dict["species"](outputs["species"], species_labels)
            loss_family = criterion_dict["family"](outputs["family"], family_labels)
            loss_order = criterion_dict["order"](outputs["order"], order_labels)

            # Weighted sum of losses (equal weights for now)
            loss = loss_species + loss_family + loss_order

        # Backward Pass with Scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion_dict, device):
    """
    Validates the model on the validation set.
    Calculates Macro F1 Score for the primary species task.
    """
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            species_labels = labels[0].to(device)
            family_labels = labels[1].to(device)
            order_labels = labels[2].to(device)

            outputs = model(images)

            loss_species = criterion_dict["species"](outputs["species"], species_labels)
            loss_family = criterion_dict["family"](outputs["family"], family_labels)
            loss_order = criterion_dict["order"](outputs["order"], order_labels)

            loss = loss_species + loss_family + loss_order
            running_loss += loss.item() * images.size(0)

            # Collect predictions for metric calculation (Species only)
            preds = torch.argmax(outputs["species"], dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(species_labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    macro_f1 = calculate_macro_f1(all_targets, all_preds)

    return epoch_loss, macro_f1


def run_training_pipeline(debug_size=Config.DEBUG_SAMPLE_SIZE):
    """
    Orchestrates the two-stage training process.

    Stage 1: Representation Learning (Instance-Balanced, Full Model)
    Stage 2: Classifier Re-balancing (Class-Balanced, Frozen Backbone)
    """
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup()
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Initialize Model
    print("Initializing HierarchicalConvNeXt model...")
    model = HierarchicalConvNeXt(pretrained=Config.PRETRAINED)
    model.to(device)

    # 3. Define Loss Functions
    # Label smoothing is applied to the primary task to prevent overconfidence
    criterion_dict = {
        "species": nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING),
        "family": nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING),
        "order": nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING),
    }

    # ====================================================
    # STAGE 1: Representation Learning
    # ====================================================
    print("\n" + "=" * 40)
    print("STAGE 1: Representation Learning")
    print("=" * 40)

    # Get Dataloaders for Stage 1 (Instance Balanced)
    train_loader_s1, val_loader, test_loader = get_dataloaders(
        stage=1, debug_size=debug_size
    )

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.STAGE1_LR, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR requires steps per epoch
    steps_per_epoch = len(train_loader_s1)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.STAGE1_LR,
        epochs=Config.STAGE1_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
    )

    scaler = GradScaler()

    best_f1 = 0.0

    for epoch in range(Config.STAGE1_EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader_s1, optimizer, criterion_dict, device, scaler, scheduler
        )
        val_loss, val_f1 = validate(model, val_loader, criterion_dict, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.STAGE1_EPOCHS} | "
            f"Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val F1: {val_f1}"
        )

        # Save Best Model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved with F1: {best_f1}")

    # ====================================================
    # STAGE 2: Classifier Re-balancing
    # ====================================================
    print("\n" + "=" * 40)
    print("STAGE 2: Classifier Re-balancing")
    print("=" * 40)

    # Load best weights from Stage 1
    print("Loading best weights from Stage 1...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # Freeze Backbone
    print("Freezing backbone...")
    model.freeze_backbone()

    # Get Dataloaders for Stage 2 (Class Balanced)
    # Note: val_loader and test_loader remain the same
    train_loader_s2, _, _ = get_dataloaders(stage=2, debug_size=debug_size)

    # Re-initialize Optimizer for Heads Only (Backbone params have requires_grad=False)
    # Use a lower learning rate for fine-tuning
    optimizer_s2 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.STAGE2_LR,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Simple scheduler or no scheduler for short fine-tuning
    scheduler_s2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_s2, T_max=Config.STAGE2_EPOCHS
    )

    # Reset Scaler
    scaler_s2 = GradScaler()

    for epoch in range(Config.STAGE2_EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader_s2,
            optimizer_s2,
            criterion_dict,
            device,
            scaler_s2,
            scheduler_s2,
        )
        val_loss, val_f1 = validate(model, val_loader, criterion_dict, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.STAGE2_EPOCHS} | "
            f"Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val F1: {val_f1}"
        )

        # Save if better
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved with F1: {best_f1}")

    # ====================================================
    # INFERENCE & SUBMISSION
    # ====================================================
    print("\n" + "=" * 40)
    print("Generating Submission")
    print("=" * 40)

    # Load absolute best model
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # Compute Class Priors for Logit Adjustment
    print("Computing class priors for Post-Hoc Logit Adjustment...")
    class_priors = compute_class_priors(load_cached_data=True)

    # Generate Submission
    print(f"Predicting on test set ({len(test_loader.dataset)} samples)...")
    generate_submission(
        model,
        test_loader,
        device,
        class_priors=class_priors,
        output_path=Config.SUBMISSION_PATH,
    )
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
