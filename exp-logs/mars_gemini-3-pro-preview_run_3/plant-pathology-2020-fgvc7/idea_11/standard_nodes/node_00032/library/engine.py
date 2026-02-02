import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import AppleDataset, get_transforms
from library.modeling import AppleNet
from library.utils import (
    seed_everything,
    calculate_roc_auc,
    compute_class_weights,
    ModelEMA,
)


def train_one_epoch(model, ema_model, loader, criterion, optimizer, scaler, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        with autocast(enabled=Config.USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if ema_model is not None:
            ema_model.update(model)

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    preds_list = []
    targets_list = []
    dataset_size = len(loader.dataset)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)

            with autocast(enabled=Config.USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, targets)

            # Apply softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)

            running_loss += loss.item() * images.size(0)
            preds_list.append(probs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    auc = calculate_roc_auc(targets, preds)

    return epoch_loss, auc, preds


def train_fold(train_df, val_df, model_name, img_size, output_name):
    """
    Trains a model for a specific fold configuration.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        model_name (str): Name of the timm backbone.
        img_size (int): Input image resolution.
        output_name (str): Filename for saving the best model checkpoint.

    Returns:
        float: Best validation AUC score.
    """
    device = Config.DEVICE
    seed_everything(Config.SEED)

    # --- Data Loaders ---
    train_dataset = AppleDataset(train_df, transforms=get_transforms("train", img_size))
    val_dataset = AppleDataset(val_df, transforms=get_transforms("valid", img_size))

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

    # --- Model Setup ---
    model = AppleNet(
        model_name=model_name, num_classes=Config.NUM_CLASSES, pretrained=True
    )
    model.to(device)

    # --- EMA Setup ---
    ema_model = None
    if Config.USE_EMA:
        ema_model = ModelEMA(model, decay=Config.EMA_DECAY)

    # --- Loss & Optimizer ---
    if Config.USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(train_df, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    scaler = GradScaler(enabled=Config.USE_AMP)

    # --- Training Loop ---
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, f"{output_name}.pth")

    print(
        f"Starting training for {output_name} with {model_name} @ {img_size}x{img_size}"
    )

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, ema_model, train_loader, criterion, optimizer, scaler, device
        )

        # Evaluate using EMA model if available, otherwise standard model
        eval_model = ema_model.ema_model if ema_model else model
        val_loss, val_auc, _ = validate(eval_model, val_loader, criterion, device)

        scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Checkpointing & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(eval_model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Cleanup
    del model, ema_model, optimizer, scaler, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    return best_auc


def inference(model_path, test_df, model_name, img_size):
    """
    Performs inference on the test set using TTA (Horizontal Flip).

    Args:
        model_path (str): Path to the trained model checkpoint.
        test_df (pd.DataFrame): Test metadata.
        model_name (str): Name of the architecture.
        img_size (int): Input resolution.

    Returns:
        np.ndarray: Predicted probabilities (N, Num_Classes).
    """
    device = Config.DEVICE

    # Load Model
    model = AppleNet(
        model_name=model_name, num_classes=Config.NUM_CLASSES, pretrained=False
    )
    # Load weights (map to device to ensure safety)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Dataset
    test_dataset = AppleDataset(test_df, transforms=get_transforms("test", img_size))

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Prediction Accumulator
    # We will average predictions from Original and Horizontal Flip
    predictions = []

    # 1. Original Pass
    preds_orig = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            with autocast(enabled=Config.USE_AMP):
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
            preds_orig.append(probs.cpu().numpy())
    predictions.append(np.concatenate(preds_orig, axis=0))

    # 2. TTA: Horizontal Flip
    if Config.TTA_HORIZONTAL_FLIP:
        preds_flip = []
        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                # Flip images horizontally (dim 3 is width: B, C, H, W)
                images_flipped = torch.flip(images, dims=[3])

                with autocast(enabled=Config.USE_AMP):
                    outputs = model(images_flipped)
                    probs = torch.softmax(outputs, dim=1)
                preds_flip.append(probs.cpu().numpy())
        predictions.append(np.concatenate(preds_flip, axis=0))

    # Average predictions
    final_preds = np.mean(predictions, axis=0)

    # Cleanup
    del model, test_loader
    torch.cuda.empty_cache()
    gc.collect()

    return final_preds
