import os
import time
import gc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from timm.utils import ModelEmaV2

from library.config import Config
from library.utils import seed_everything, AverageMeter, get_score
from library.dataset import CatheterDataset, get_transforms
from library.model import CatheterModel


def train_one_epoch(
    model, loader, optimizer, scheduler, criterion, device, scaler, model_ema=None
):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass with Mixed Precision
        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        # Backward pass with Scaler
        scaler.scale(loss).backward()

        # Update weights
        scaler.step(optimizer)
        scaler.update()

        # Update EMA (Cite solution_lesson_node_00014)
        if model_ema is not None:
            model_ema.update(model)

        # Update scheduler
        scheduler.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def valid_one_epoch(model, loader, criterion, device):
    """
    Validates the model on the validation set.
    Returns average loss and AUC score.
    """
    model.eval()
    losses = AverageMeter()

    # Store predictions and targets for AUC calculation
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            losses.update(loss.item(), images.size(0))

            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(logits)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate AUC
    score = get_score(all_targets, all_preds)

    return losses.avg, score


def run_training():
    """
    Main function to run the training pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    device = Config.DEVICE

    print(f"Device: {device}")
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Image Size: {Config.IMAGE_SIZE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")

    # 2. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    df_val = pd.read_csv(Config.VAL_METADATA)

    # Debug Mode
    if Config.DEBUG:
        print(f"DEBUG mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        df_train = df_train.sample(
            min(len(df_train), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        df_val = df_val.sample(
            min(len(df_val), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    print(f"Train Size: {len(df_train)}")
    print(f"Val Size: {len(df_val)}")

    # 3. Datasets and DataLoaders
    train_dataset = CatheterDataset(
        df_train, transforms=get_transforms(data="train"), mode="train"
    )
    val_dataset = CatheterDataset(
        df_val,
        transforms=get_transforms(data="valid"),
        mode="train",  # 'train' mode returns labels
    )

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
        drop_last=False,
    )

    # 4. Model Initialization
    model = CatheterModel(pretrained=Config.PRETRAINED)
    model.to(device)

    # 5. Optimization
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Model EMA
    model_ema = None
    if Config.USE_EMA:
        print("Initializing Model EMA...")
        model_ema = ModelEmaV2(model, decay=Config.EMA_DECAY)

    # BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()

    # Gradient Scaler for AMP
    scaler = torch.cuda.amp.GradScaler()

    # Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    # 6. Training Loop
    best_score = -np.inf
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            scaler,
            model_ema,
        )

        # Validate (Use EMA model if available)
        eval_model = model_ema.module if model_ema is not None else model
        val_loss, val_score = valid_one_epoch(eval_model, val_loader, criterion, device)

        elapsed = time.time() - start_time
        cur_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1}/{Config.EPOCHS} | Time: {elapsed:.1f}s")
        print(f"Train Loss: {train_loss} | Val Loss: {val_loss}")
        print(f"Val AUC: {val_score} | LR: {cur_lr}")

        # Checkpointing and Early Stopping
        if val_score > best_score:
            print(f"Score Improved ({best_score} -> {val_score}). Saving model...")
            best_score = val_score
            save_model = model_ema.module if model_ema is not None else model
            torch.save(save_model.state_dict(), Config.BEST_MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val AUC: {best_score}")

    # Cleanup
    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()
