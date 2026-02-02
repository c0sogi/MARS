import os
import numpy as np
import torch
import torch.nn as nn
from library.config import Config
from library.utils import AverageMeter, get_score


def train_one_epoch(model, optimizer, scheduler, dataloader, device, scaler):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    for step, data in enumerate(dataloader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["labels"].to(device)

        batch_size = input_ids.size(0)

        # Mixed Precision Forward Pass
        with torch.cuda.amp.autocast(enabled=Config.use_fp16):
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            # Normalize loss for gradient accumulation
            if Config.accumulate_grad_batches > 1:
                loss = loss / Config.accumulate_grad_batches

        # Backward Pass
        scaler.scale(loss).backward()

        # Optimizer Step (with Gradient Accumulation)
        if (step + 1) % Config.accumulate_grad_batches == 0:
            # Unscale before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            # Step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Scheduler Step (OneCycleLR steps per batch)
            if scheduler is not None:
                scheduler.step()

        # Update metrics (multiply back by accumulation factor to log true loss)
        losses.update(loss.item() * Config.accumulate_grad_batches, batch_size)

    return losses.avg


def valid_one_epoch(model, dataloader, device):
    """
    Validates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    preds = []
    targets = []

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            labels = data["labels"].to(device)

            batch_size = input_ids.size(0)

            with torch.cuda.amp.autocast(enabled=Config.use_fp16):
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

            losses.update(loss.item(), batch_size)

            # Apply sigmoid for probabilities
            batch_preds = torch.sigmoid(outputs).detach().cpu().numpy()
            batch_targets = labels.detach().cpu().numpy()

            preds.append(batch_preds)
            targets.append(batch_targets)

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    # Calculate Metric
    score = get_score(targets, preds)

    return losses.avg, score, preds


def predict(model, dataloader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)

            with torch.cuda.amp.autocast(enabled=Config.use_fp16):
                outputs = model(input_ids, attention_mask)

            batch_preds = torch.sigmoid(outputs).detach().cpu().numpy()
            preds.append(batch_preds)

    preds = np.concatenate(preds, axis=0)
    return preds


def train_loop(
    model, train_loader, val_loader, optimizer, scheduler, device, patience=2
):
    """
    Orchestrates the training process with Early Stopping and Model Checkpointing.
    """
    best_score = -np.inf
    scaler = torch.cuda.amp.GradScaler(enabled=Config.use_fp16)

    # Early stopping counter
    patience_counter = 0

    print(f"Starting training for {Config.epochs} epochs on {device}...")

    for epoch in range(Config.epochs):
        # Train
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, scaler
        )

        # Validate
        val_loss, val_score, _ = valid_one_epoch(model, val_loader, device)

        # Print Metrics (Full Precision)
        print(f"Epoch {epoch+1}/{Config.epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss:   {val_loss}")
        print(f"Val AUC:    {val_score}")

        # Checkpointing & Early Stopping
        if val_score > best_score:
            print(
                f"Validation Score Improved ({best_score} -> {val_score}). Saving model..."
            )
            best_score = val_score
            torch.save(model.state_dict(), Config.model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"Validation Score did not improve. Patience: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print("Early stopping triggered. Stopping training.")
            break

    print(f"Training complete. Best Val AUC: {best_score}")

    # Load best model for future use
    model.load_state_dict(torch.load(Config.model_save_path, map_location=device))
    return model
