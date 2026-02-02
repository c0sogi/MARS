import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from library.config import Config
from library.utils import seed_everything, get_class_weights, calculate_roc_auc
from library.data import get_loaders, prepare_folds
from library.models import AppleEfficientNet, AppleSwin


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Executes one training epoch using Mixed Precision (AMP).
    """
    model.train()
    running_loss = 0.0
    y_true_list = []
    y_pred_list = []

    scaler = GradScaler()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            # labels are one-hot encoded. Convert to indices for CrossEntropyLoss
            target_indices = labels.argmax(dim=1).long()
            loss = criterion(outputs, target_indices)

        scaler.scale(loss).backward()

        # Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        # Store predictions (probs) and true labels (one-hot) for AUC
        y_true_list.append(labels.detach().cpu().numpy())
        y_pred_list.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    epoch_auc = calculate_roc_auc(y_true, y_pred)

    return epoch_loss, epoch_auc


def validate_one_epoch(model, loader, criterion, device):
    """
    Executes validation for one epoch using Mixed Precision (AMP).
    """
    model.eval()
    running_loss = 0.0
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            with autocast():
                outputs = model(images)
                target_indices = labels.argmax(dim=1).long()
                loss = criterion(outputs, target_indices)

            running_loss += loss.item() * images.size(0)

            y_true_list.append(labels.cpu().numpy())
            y_pred_list.append(torch.softmax(outputs, dim=1).float().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    epoch_auc = calculate_roc_auc(y_true, y_pred)

    return epoch_loss, epoch_auc


def run_fold(fold, model_type, debug=False):
    """
    Trains a specific model architecture for a specific fold.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(f"\n{'='*30}")
    print(f"Fold {fold} | Model: {model_type}")
    print(f"{'='*30}")

    # 1. Initialize Model
    if model_type == "effnet":
        model = AppleEfficientNet(pretrained=True)
        img_size = Config.EFFNET_IMG_SIZE
        save_name = f"effnet_fold_{fold}_best.pth"
    elif model_type == "swin":
        model = AppleSwin(pretrained=True)
        img_size = Config.SWIN_IMG_SIZE
        save_name = f"swin_fold_{fold}_best.pth"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model.to(device)

    # 2. Get DataLoaders
    train_loader, val_loader = get_loaders(
        fold, img_size, Config.BATCH_SIZE, debug=debug
    )

    # 3. Compute Class Weights for this fold's training set
    # We load the full dataframe and filter out the validation fold to get the training set
    full_df = prepare_folds(load_cached_data=True)
    if debug:
        # In debug mode, get_loaders samples the df. We attempt to approximate or just use full df weights.
        train_df = full_df[full_df["fold"] != fold]
    else:
        train_df = full_df[full_df["fold"] != fold]

    # Force recompute to ensure weights match the specific fold split
    class_weights = get_class_weights(train_df, load_cached_data=False)

    # 4. Setup Training Components
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0
    save_path = os.path.join(Config.WORKING_DIR, save_name)

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate_one_epoch(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - start_time

        # Print metrics with full precision
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | Time: {elapsed:.2f}s")
        print(f"  Train Loss: {train_loss} | Train AUC: {train_auc}")
        print(f"  Val Loss:   {val_loss} | Val AUC:   {val_auc}")

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> New Best AUC found. Model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"  -> Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Fold {fold} {model_type} finished. Best Val AUC: {best_auc}")

    # Cleanup to free GPU memory
    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()


def run_training(debug=Config.DEBUG):
    """
    Orchestrates the training of the entire ensemble (All Folds x All Models).
    """
    print(f"Starting Training Pipeline. Debug Mode: {debug}")
    print(f"Device: {Config.DEVICE}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    for fold in range(Config.N_FOLDS):
        # Train EfficientNet-B4
        run_fold(fold, "effnet", debug=debug)

        # Train Swin-Small
        run_fold(fold, "swin", debug=debug)

    print("All training runs completed successfully.")
