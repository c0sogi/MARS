import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from library.config import Config
from library.utils import (
    seed_everything,
    AverageMeter,
    LaplaceLogLikelihoodLoss,
    calculate_metric_numpy,
)
from library.data import PulmonaryDataset
from library.model import TriSlabModel


def train_epoch(loader, model, optimizer, loss_fn, device):
    """
    Performs one epoch of training.
    """
    model.train()
    loss_meter = AverageMeter()

    for batch_idx, (imgs, tabular, base_fvc, time_delta, targets) in enumerate(loader):
        # Move data to device
        imgs = imgs.to(device)
        tabular = tabular.to(device)
        base_fvc = base_fvc.to(device)
        time_delta = time_delta.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass: Predict parameters [alpha, sigma_base, sigma_growth]
        preds = model(imgs, tabular)

        # Calculate loss
        loss = loss_fn(preds, targets, base_fvc, time_delta)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_meter.update(loss.item(), imgs.size(0))

    return loss_meter.avg


def validate_epoch(loader, model, loss_fn, device):
    """
    Performs validation and calculates the competition metric.
    """
    model.eval()
    loss_meter = AverageMeter()
    metric_meter = AverageMeter()

    with torch.no_grad():
        for batch_idx, (imgs, tabular, base_fvc, time_delta, targets) in enumerate(
            loader
        ):
            imgs = imgs.to(device)
            tabular = tabular.to(device)
            base_fvc = base_fvc.to(device)
            time_delta = time_delta.to(device)
            targets = targets.to(device)

            # Forward pass
            preds = model(imgs, tabular)

            # Calculate Loss (Negative Log Likelihood proxy)
            loss = loss_fn(preds, targets, base_fvc, time_delta)
            loss_meter.update(loss.item(), imgs.size(0))

            # Calculate Actual Metric (Laplace Log Likelihood)
            # Move tensors to CPU numpy for metric calculation
            preds_np = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()
            base_fvc_np = base_fvc.cpu().numpy()
            time_delta_np = time_delta.cpu().numpy()

            score = calculate_metric_numpy(
                preds_np, targets_np, base_fvc_np, time_delta_np
            )
            metric_meter.update(score, imgs.size(0))

    return loss_meter.avg, metric_meter.avg


def run_training(debug=False):
    """
    Main orchestration function for training the model.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    if debug or Config.DEBUG:
        print(
            f"DEBUG MODE: limiting training data to {Config.DEBUG_SAMPLE_SIZE} samples."
        )
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Define Augmentations (Spatial only, Cite solution_lesson_node_00005)
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5,
            ),
            A.Normalize(mean=0, std=1, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Normalize(mean=0, std=1, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    train_dataset = PulmonaryDataset(train_df, mode="train", transform=train_transform)
    val_dataset = PulmonaryDataset(val_df, mode="val", transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = TriSlabModel(Config)
    model = model.to(device)

    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN)

    loss_fn = LaplaceLogLikelihoodLoss()

    # 4. Training Loop
    best_metric = -float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_epoch(train_loader, model, optimizer, loss_fn, device)

        # Validate
        val_loss, val_metric = validate_epoch(val_loader, model, loss_fn, device)

        # Step Scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Metric: {val_metric:.10f}"
        )

        # Early Stopping Logic
        # Metric is negative and higher is better (e.g., -6.5 is better than -6.8)
        if val_metric > best_metric:
            print(
                f"Metric improved from {best_metric:.10f} to {val_metric:.10f}. Saving model..."
            )
            best_metric = val_metric
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Metric: {best_metric:.10f}")
