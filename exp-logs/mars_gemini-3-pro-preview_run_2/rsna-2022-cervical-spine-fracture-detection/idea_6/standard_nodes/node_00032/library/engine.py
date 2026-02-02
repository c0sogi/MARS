import os
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from library.config import Config
from library.dataset import CervicalSpineDataset
from library.model import CervicalSpineTransformer
from library.loss import WeightedMultiLabelLogLoss
from library.utils import seed_everything


def train_one_epoch(
    model, loader, optimizer, criterion, device, scaler, scheduler=None
):
    """
    Trains the model for one epoch using Mixed Precision and Gradient Accumulation.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    accumulation_steps = Config.ACCUMULATION_STEPS
    optimizer.zero_grad()

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)

        # Mixed Precision Forward Pass
        with autocast():
            logits = model(images)
            loss = criterion(logits, targets)
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps

        # Backward pass with scaler
        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Scheduler step (if per-step scheduler)
            if scheduler:
                scheduler.step()

        # Update metrics (multiply back by accumulation steps to get true loss)
        running_loss += loss.item() * accumulation_steps * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)

            # Standard forward pass (AMP optional for inference, usually not needed for loss calc)
            logits = model(images)
            loss = criterion(logits, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def predict_test_set(model, loader, device):
    """
    Generates predictions for the test set and formats them for submission.
    """
    model.eval()
    predictions = []

    # Column names corresponding to the 8 output neurons
    col_names = [f"C{i}" for i in range(1, 8)] + ["patient_overall"]

    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = images.to(device)
            batch_size = images.size(0)

            # Retrieve StudyInstanceUIDs for the current batch
            # Assumes loader is not shuffled
            start_idx = i * loader.batch_size
            end_idx = start_idx + batch_size
            batch_uids = loader.dataset.df.iloc[start_idx:end_idx][
                "StudyInstanceUID"
            ].values

            # Inference
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            # Format rows
            for uid, prob_row in zip(batch_uids, probs):
                for col_name, prob in zip(col_names, prob_row):
                    row_id = f"{uid}_{col_name}"
                    predictions.append({"row_id": row_id, "fractured": prob})

    return pd.DataFrame(predictions)


def run(epochs=Config.EPOCHS, debug=Config.DEBUG):
    """
    Main execution function: sets up data, model, training loop, and inference.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Ensure output directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # --- Data Loading ---
    train_dataset = CervicalSpineDataset(split="train", debug=debug)
    val_dataset = CervicalSpineDataset(split="val", debug=debug)
    test_dataset = CervicalSpineDataset(split="test", debug=debug)

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Model & Training Setup ---
    model = CervicalSpineTransformer().to(device)

    # Loss: Weighted Multi-Label Log Loss
    # We use the custom loss which handles class imbalance internally
    criterion = WeightedMultiLabelLogLoss(pos_weight=Config.POS_WEIGHT).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Cosine Annealing
    # T_max is total number of steps
    total_steps = epochs * len(train_loader) // Config.ACCUMULATION_STEPS
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=Config.ETA_MIN,
    )

    # Gradient Scaler for AMP
    scaler = GradScaler()

    # --- Training Loop ---
    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = 3
    counter = 0

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, scheduler
        )
        val_loss = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{epochs} | Time: {elapsed:.0f}s")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), Config.CHECKPOINT_PATH)
            print(f"New best model saved to {Config.CHECKPOINT_PATH}")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    # --- Inference ---
    print("Loading best model for inference...")
    model.load_state_dict(best_model_wts)

    print("Generating predictions on test set...")
    submission_df = predict_test_set(model, test_loader, device)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
