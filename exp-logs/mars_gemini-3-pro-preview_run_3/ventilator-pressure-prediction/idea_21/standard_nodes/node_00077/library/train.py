import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from library.config import Config
from library.model import RDHNet
from library.data import load_and_preprocess
from library.utils import seed_everything, AverageMeter, compute_mae, Timer, log_metrics


def train_epoch(model, loader, optimizer, scheduler, device, max_grad_norm):
    """
    Executes one training epoch.
    Computes Masked L1 Loss on the inspiratory phase and updates model weights.
    """
    model.train()
    loss_meter = AverageMeter()

    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        u_out = batch["u_out"].to(device)

        optimizer.zero_grad()
        preds = model(inputs)

        # Masked L1 Loss (MAE)
        # We only calculate loss where u_out == 0 (Inspiratory phase)
        mask = (u_out == 0).float()
        # Add epsilon to denominator to prevent division by zero
        loss = (torch.abs(preds - targets) * mask).sum() / (mask.sum() + 1e-8)

        loss.backward()

        # Gradient Clipping to stabilize LSTM/Hybrid training
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()

        loss_meter.update(loss.item(), inputs.size(0))

    return loss_meter.avg


def validate_epoch(model, loader, device):
    """
    Executes one validation epoch.
    Computes Loss and MAE metrics without updating gradients.
    """
    model.eval()
    loss_meter = AverageMeter()
    mae_meter = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            u_out = batch["u_out"].to(device)

            preds = model(inputs)

            # Masked L1 Loss
            mask = (u_out == 0).float()
            loss = (torch.abs(preds - targets) * mask).sum() / (mask.sum() + 1e-8)

            # Metric: MAE (using library utility)
            mae = compute_mae(preds, targets, u_out)

            loss_meter.update(loss.item(), inputs.size(0))
            mae_meter.update(mae, inputs.size(0))

    return loss_meter.avg, mae_meter.avg


def generate_submission(model, loader, device, output_path, cache_dir):
    """
    Generates predictions for the test set using the best model and saves to CSV.
    """
    print("Generating submission file...")
    model.eval()
    preds_list = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            preds = model(inputs)
            preds_list.append(preds.cpu().numpy())

    # Concatenate predictions: (N_batches, B, 80) -> (N_total, 80)
    all_preds = np.concatenate(preds_list, axis=0)

    # Flatten to match sample submission format (Row-wise time steps)
    flat_preds = all_preds.flatten()

    # Load test IDs from cache (saved during preprocessing)
    test_ids_path = os.path.join(cache_dir, "test_ids.npy")
    if not os.path.exists(test_ids_path):
        raise FileNotFoundError(f"Test IDs not found at {test_ids_path}")

    test_ids = np.load(test_ids_path)

    # Sanity check
    if len(test_ids) != len(flat_preds):
        print(
            f"Warning: Length mismatch. IDs: {len(test_ids)}, Preds: {len(flat_preds)}"
        )

    # Create DataFrame
    submission = pd.DataFrame({"id": test_ids, "pressure": flat_preds})

    # Save
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    max_grad_norm=Config.MAX_GRAD_NORM,
    warmup_epochs=Config.WARMUP_EPOCHS,
    cosine_cycles=Config.COSINE_CYCLES,
    patience=Config.EARLY_STOPPING_PATIENCE,
    debug=Config.DEBUG,
):
    """
    Main pipeline execution function.
    Handles setup, data loading, model training, validation, and submission generation.
    """
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Apply debug flag if provided
    Config.DEBUG = debug

    # 2. Data Loading
    print("Initializing data pipeline...")
    # Load cached data or compute from scratch
    train_ds, val_ds, test_ds = load_and_preprocess(load_cached_data=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing RDH-Net...")
    model = RDHNet().to(device)

    # 4. Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Scheduler Setup
    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = len(train_loader) * warmup_epochs

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=cosine_cycles,
    )

    # 5. Training Loop
    print(f"Starting training for {epochs} epochs on {device}...")
    best_val_mae = float("inf")
    early_stop_counter = 0

    for epoch in range(1, epochs + 1):
        with Timer(f"Epoch {epoch}") as timer:
            # Train
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, device, max_grad_norm
            )

            # Validate
            val_loss, val_mae = validate_epoch(model, val_loader, device)

        # Log Metrics
        log_metrics(epoch, train_loss, val_loss, val_mae, timer.end - timer.start)

        # Checkpoint & Early Stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            early_stop_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved! MAE: {best_val_mae}")
        else:
            early_stop_counter += 1
            print(
                f"No improvement. Early stopping counter: {early_stop_counter}/{patience}"
            )

        if early_stop_counter >= patience:
            print("Early stopping triggered. Stopping training.")
            break

    # 6. Submission
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    generate_submission(
        model, test_loader, device, Config.SUBMISSION_PATH, Config.CACHE_DIR
    )


# Execute the training pipeline
run_training()
