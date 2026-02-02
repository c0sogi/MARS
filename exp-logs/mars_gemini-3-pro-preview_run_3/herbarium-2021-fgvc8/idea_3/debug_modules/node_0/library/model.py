import os
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import numpy as np
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from library.config import Config
from library.utils import calculate_f1_macro


class HerbariumConvNeXt(nn.Module):
    """
    ConvNeXt-Small based model for Herbarium Plant Species Classification.
    Uses a pretrained backbone with a reduced embedding dimension (768)
    mapped to 64,500 classes via a single linear layer.
    """

    def __init__(self, pretrained=True):
        super(HerbariumConvNeXt, self).__init__()

        # Initialize backbone
        # num_classes=0 ensures we get the pooled feature vector (embedding)
        # global_pool is handled by timm internally when num_classes=0
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=Config.DROP_PATH_RATE,
        )

        # Classification Head
        # Maps 768-dim embedding to 64,500 classes
        self.head = nn.Linear(Config.EMBEDDING_DIM, Config.NUM_CLASSES)

    def forward(self, x):
        # Extract features (B, 768)
        features = self.backbone(x)

        # Predict logits (B, 64500)
        logits = self.head(features)

        return logits


def train_one_epoch(
    model, loader, optimizer, criterion, device, scaler, scheduler=None
):
    """
    Trains the model for one epoch using Mixed Precision.
    """
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        # Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and Macro F1 score.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # Get predictions
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    macro_f1 = calculate_f1_macro(all_labels, all_preds)

    return epoch_loss, macro_f1


def train_model(train_loader, val_loader):
    """
    Main training pipeline implementing:
    - AdamW Optimizer
    - CrossEntropy with Label Smoothing
    - Cosine Annealing Scheduler
    - Stochastic Weight Averaging (SWA)
    - Early Stopping
    """
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Initialize Model
    model = HerbariumConvNeXt(pretrained=True)
    model = model.to(device)

    # Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    scaler = GradScaler()

    # Scheduler (Cosine Annealing for main phase)
    # We aim to reach MIN_LR before SWA starts
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.SWA_START_EPOCH, eta_min=Config.MIN_LR
    )

    # SWA Setup
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR)

    best_f1 = 0.0
    patience = 3
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        current_epoch = epoch + 1

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )

        # SWA Logic
        if Config.USE_SWA and current_epoch > Config.SWA_START_EPOCH:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            lr = swa_scheduler.get_last_lr()[0]
            print(
                f"Epoch {current_epoch}/{Config.EPOCHS} [SWA] - LR: {lr:.2e} - Train Loss: {train_loss:.6f}"
            )
        else:
            # Regular Validation
            val_loss, val_f1 = validate(model, val_loader, criterion, device)
            main_scheduler.step()
            lr = main_scheduler.get_last_lr()[0]

            print(
                f"Epoch {current_epoch}/{Config.EPOCHS} - LR: {lr:.2e} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val F1: {val_f1:.16f}"
            )

            # Save Best Model (based on regular validation)
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"  New best model saved! F1: {best_f1:.16f}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early Stopping (only applies before SWA takes over fully)
            if patience_counter >= patience and not Config.USE_SWA:
                print("Early stopping triggered.")
                break

    # Finalize SWA
    if Config.USE_SWA:
        print("Finalizing SWA model (updating BatchNorm statistics)...")
        update_bn(train_loader, swa_model, device=device)
        torch.save(swa_model.state_dict(), Config.SWA_MODEL_SAVE_PATH)
        print(f"SWA model saved to {Config.SWA_MODEL_SAVE_PATH}")

        # Optional: Validate SWA model
        swa_loss, swa_f1 = validate(swa_model, val_loader, criterion, device)
        print(f"SWA Model Validation - Loss: {swa_loss:.6f} - F1: {swa_f1:.16f}")

        return swa_model
    else:
        # Load best model for return
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
        return model


def generate_submission(model, test_loader):
    """
    Generates predictions for the test set and saves to CSV.
    """
    device = torch.device(Config.DEVICE)
    model.eval()
    model.to(device)

    predictions = []
    ids = []

    print("Generating submission...")

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device, non_blocking=True)

            with autocast():
                outputs = model(images)

            preds = torch.argmax(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            ids.extend(image_ids.numpy())

    # Create DataFrame
    df_sub = pd.DataFrame({"Id": ids, "Predicted": predictions})

    # Sort by Id to match sample submission format
    df_sub = df_sub.sort_values("Id").reset_index(drop=True)

    # Save
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Total predictions: {len(df_sub)}")
