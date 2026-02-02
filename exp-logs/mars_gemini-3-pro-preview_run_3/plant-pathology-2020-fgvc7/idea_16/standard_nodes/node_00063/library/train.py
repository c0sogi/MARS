import os
import time
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import (
    seed_everything,
    compute_class_weights,
    ModelEMA,
    calculate_metric,
)
from library.dataset import get_loaders, load_data
from library.models import AppleNet


def train_one_epoch(
    model, loader, optimizer, scheduler, criterion, device, scaler, ema_model=None
):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        with autocast(enabled=Config.USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        if ema_model and (batch_idx % Config.EMA_UPDATE_EVERY == 0):
            ema_model.update(model)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Validates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # Apply softmax for metric calculation
            probs = torch.softmax(outputs, dim=1)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            preds_list.append(probs.cpu())
            targets_list.append(labels.cpu())

    epoch_loss = running_loss / dataset_size

    preds = torch.cat(preds_list, dim=0)
    targets = torch.cat(targets_list, dim=0)

    score = calculate_metric(targets, preds)

    return epoch_loss, score, preds.numpy()


def inference_fn(model, loader, device):
    """
    Generates predictions with Test-Time Augmentation (TTA).
    TTA Strategy: Original, HFlip, VFlip, Transpose.
    """
    model.eval()
    preds_list = []

    with torch.no_grad():
        for images in loader:
            images = images.to(device)

            # 1. Original
            out_1 = model(images)

            if Config.USE_TTA:
                # 2. Horizontal Flip
                out_2 = model(torch.flip(images, [3]))

                # 3. Vertical Flip
                out_3 = model(torch.flip(images, [2]))

                # 4. Transpose (Rotate 90 + HFlip equivalent, or just transpose H/W)
                # Transpose last two dims: (B, C, W, H)
                out_4 = model(torch.transpose(images, 2, 3))

                # Average logits (or probabilities)
                # Averaging probabilities is generally safer for ensembles
                p1 = torch.softmax(out_1, dim=1)
                p2 = torch.softmax(out_2, dim=1)
                p3 = torch.softmax(out_3, dim=1)
                p4 = torch.softmax(out_4, dim=1)

                avg_probs = (p1 + p2 + p3 + p4) / 4.0
                preds_list.append(avg_probs.cpu())
            else:
                preds_list.append(torch.softmax(out_1, dim=1).cpu())

    return torch.cat(preds_list, dim=0).numpy()


def run_training():
    """
    Main orchestration function.
    """
    seed_everything(Config.SEED)

    # 1. Load Data
    # We use the provided get_loaders which handles caching internally via load_data
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # Need dataframe for class weights
    train_df, _, _ = load_data(load_cached_data=True)

    # 2. Compute Class Weights
    class_weights = compute_class_weights(train_df, load_cached_data=True)
    print(f"Class Weights: {class_weights}")

    # 3. Setup Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Store predictions from all models for ensemble
    ensemble_preds = []

    # 4. Iterate over Backbones (Heterogeneous Ensemble)
    for i, backbone_name in enumerate(Config.BACKBONES):
        print(f"\n{'='*40}")
        print(f"Training Model {i+1}/{len(Config.BACKBONES)}: {backbone_name}")
        print(f"{'='*40}")

        # Initialize Model
        model = AppleNet(backbone_name, Config.NUM_CLASSES, pretrained=True)
        model.to(Config.DEVICE)

        # Initialize EMA
        ema = (
            ModelEMA(model, decay=Config.EMA_DECAY, device=Config.DEVICE)
            if Config.USE_EMA
            else None
        )

        # Optimizer & Scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Steps per epoch
        num_steps = len(train_loader) * Config.EPOCHS
        scheduler = CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=Config.MIN_LR)

        scaler = GradScaler(enabled=Config.USE_AMP)

        # Training Loop
        best_score = -np.inf
        best_loss = np.inf
        patience_counter = 0
        best_model_path = os.path.join(Config.WORKING_DIR, f"{backbone_name}_best.pth")

        for epoch in range(Config.EPOCHS):
            start_time = time.time()

            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                criterion,
                Config.DEVICE,
                scaler,
                ema,
            )

            # Validate with EMA model if available, else regular model
            val_model = ema.module if ema else model
            val_loss, val_score, _ = validate(
                val_model, val_loader, criterion, Config.DEVICE
            )

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val AUC: {val_score}"
            )

            # Early Stopping Check
            # We prioritize AUC score
            if val_score > best_score:
                best_score = val_score
                best_loss = val_loss
                patience_counter = 0
                torch.save(val_model.state_dict(), best_model_path)
                print(f"  -> New Best AUC! Model Saved.")
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        # 5. Inference
        print(f"Loading best model for inference: {backbone_name}")
        model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
        model.to(Config.DEVICE)

        preds = inference_fn(model, test_loader, Config.DEVICE)
        ensemble_preds.append(preds)

        # Cleanup
        del model, ema, optimizer, scheduler, scaler
        gc.collect()
        torch.cuda.empty_cache()

    # 6. Ensemble & Submission
    print("\nGenerating Submission...")
    if not ensemble_preds:
        raise RuntimeError("No predictions generated.")

    # Average predictions
    final_preds = np.mean(ensemble_preds, axis=0)

    # Load test df to get image_ids
    # Use the dataframe from the loader to ensure consistency with debug sampling
    test_df = test_loader.dataset.df

    # Create submission DataFrame
    submission = pd.DataFrame()
    submission["image_id"] = test_df["image_id"]

    # Assign probabilities to label columns
    for i, label in enumerate(Config.LABELS):
        submission[label] = final_preds[:, i]

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission.head())
