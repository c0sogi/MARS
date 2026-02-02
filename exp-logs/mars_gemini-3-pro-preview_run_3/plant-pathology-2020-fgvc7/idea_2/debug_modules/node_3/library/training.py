import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from library.config import (
    MODEL_EFFICIENTNET,
    MODEL_CONVNEXT,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    SWA_START_EPOCH,
    SWA_LR,
    PATIENCE,
    CACHE_DIR,
    TRAIN_CSV,
    NUM_CLASSES,
    seed_everything,
    SEED,
)
from library.utils import calculate_class_weights
from library.data import get_dataloaders
from library.models import get_model


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate_one_epoch(model, loader, criterion, device):
    """
    Validates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # Apply softmax to get probabilities for AUC
            probs = torch.softmax(outputs, dim=1)

            all_targets.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    total_loss = running_loss / len(loader.dataset)

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Calculate Macro ROC AUC (Mean column-wise)
    # We use multi_class='ovr' for multiclass classification
    try:
        # Check if we have all classes in the validation batch to avoid sklearn errors
        # In a robust pipeline, we assume validation set is stratified.
        roc_auc = roc_auc_score(
            all_targets, all_preds, multi_class="ovr", average="macro"
        )
    except ValueError:
        # Fallback if a class is missing in the small validation batch (e.g. debugging)
        roc_auc = 0.5

    return total_loss, roc_auc


def run_training(model_name, use_swa=False, debug=False, save_name="best_model.pth"):
    """
    Main execution function to train a specific model.

    Args:
        model_name (str): Name of the model architecture.
        use_swa (bool): Whether to apply Stochastic Weight Averaging.
        debug (bool): If True, runs on a subset of data.
        save_name (str): Filename to save the best model state.

    Returns:
        model: The trained PyTorch model (or SWA model).
        history: Dictionary containing training history.
    """
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Starting training for {model_name} on {device}. SWA={use_swa}, Debug={debug}"
    )

    # 1. Data Loading
    train_loader, val_loader, _ = get_dataloaders(debug=debug, batch_size=BATCH_SIZE)

    # 2. Class Weights
    # Load train metadata to compute weights
    train_df = pd.read_csv(TRAIN_CSV)
    class_weights = calculate_class_weights(train_df, device=device)

    # 3. Model Initialization
    model = get_model(model_name, pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(device)

    # 4. Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer selection based on strategy
    if model_name == MODEL_CONVNEXT:
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    else:
        # EfficientNet usually works well with standard Adam or RMSProp
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Standard Scheduler (Cosine Annealing)
    # Used for the standard training phase or if SWA is disabled
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # SWA Setup
    swa_model = None
    swa_scheduler = None
    if use_swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)

    # Training Loop Variables
    best_roc_auc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    save_path = os.path.join(CACHE_DIR, save_name)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # SWA Logic vs Standard Scheduler
        if use_swa and epoch >= SWA_START_EPOCH:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            print(f"  [SWA] Updated SWA parameters and stepped SWA scheduler.")
        else:
            scheduler.step()

        # Validation
        # If using SWA and we are in the SWA phase, we *could* validate the SWA model,
        # but typically we monitor the base model or wait until the end.
        # For consistency in reporting, we validate the current base model weights here.
        val_loss, val_auc = validate_one_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        print(f"  Train Loss: {train_loss}")
        print(f"  Val Loss: {val_loss}")
        print(f"  Val AUC: {val_auc}")

        # Early Stopping & Checkpointing
        # Note: If SWA is enabled, we usually run until the end to collect enough averages.
        # However, we still save the best base model in case SWA fails or performs worse.
        if val_auc > best_roc_auc:
            best_roc_auc = val_auc
            patience_counter = 0
            # Save the base model
            torch.save(model.state_dict(), save_path)
            print(f"  New best model saved with AUC: {best_roc_auc}")
        else:
            patience_counter += 1

        if not use_swa and patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Finalization
    if use_swa:
        print("\nFinalizing SWA Model...")
        # Update BatchNorm statistics for the SWA model
        update_bn(train_loader, swa_model, device=device)

        # Validate SWA Model
        swa_loss, swa_auc = validate_one_epoch(swa_model, val_loader, criterion, device)
        print(f"SWA Model Results -> Val Loss: {swa_loss}, Val AUC: {swa_auc}")

        # Save SWA model
        swa_save_path = save_path.replace(".pth", "_swa.pth")
        torch.save(swa_model.state_dict(), swa_save_path)
        print(f"SWA model saved to {swa_save_path}")

        return swa_model, history
    else:
        # Reload best model
        print(f"\nLoading best model from {save_path}")
        model.load_state_dict(torch.load(save_path, map_location=device))
        return model, history
