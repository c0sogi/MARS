import os
import gc
import numpy as np
import torch
import torch.nn as nn
from library.config import Config
from library.utils import compute_qwk


def train_fn(model, dataloader, optimizer, scheduler, device, scaler):
    """
    Executes one training epoch with Mixed Precision and Gradient Accumulation.

    Args:
        model: The PyTorch model to train.
        dataloader: The training DataLoader.
        optimizer: The optimizer instance.
        scheduler: The learning rate scheduler.
        device: The device to run training on (cuda/cpu).
        scaler: The GradScaler for mixed precision.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    count = 0

    # Use MSE Loss for regression
    criterion = nn.MSELoss()

    for step, batch in enumerate(dataloader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass with Mixed Precision
        with torch.amp.autocast("cuda", enabled=Config.use_amp):
            outputs = model(input_ids, attention_mask)
            logits = outputs["logits"].squeeze(-1)
            loss = criterion(logits, labels)

        # Normalize loss for gradient accumulation
        loss = loss / Config.gradient_accumulation_steps

        # Backward pass (scaled)
        scaler.scale(loss).backward()

        # Optimizer Step (only after accumulation steps)
        if (step + 1) % Config.gradient_accumulation_steps == 0:
            # Unscale gradients to allow for clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            # Update weights
            scaler.step(optimizer)
            scaler.update()

            # Clear gradients
            optimizer.zero_grad()

            # Step scheduler
            if scheduler is not None:
                scheduler.step()

        # Accumulate total loss (rescale back for logging)
        total_loss += loss.item() * Config.gradient_accumulation_steps
        count += 1

    return total_loss / count


def eval_fn(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: The validation DataLoader.
        device: The device to run evaluation on.

    Returns:
        tuple: (average_loss, predictions, true_labels)
    """
    model.eval()
    total_loss = 0.0
    count = 0

    preds = []
    true_labels = []

    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            with torch.amp.autocast("cuda", enabled=Config.use_amp):
                outputs = model(input_ids, attention_mask)
                logits = outputs["logits"].squeeze(-1)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            count += 1

            # Store predictions and labels
            preds.extend(logits.detach().float().cpu().numpy())
            true_labels.extend(labels.detach().float().cpu().numpy())

    return total_loss / count, np.array(preds), np.array(true_labels)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    patience=3,
):
    """
    Manages the full training loop, including early stopping and checkpointing.

    Args:
        model: The PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        scheduler: Scheduler.
        device: Device.
        num_epochs: Maximum number of epochs.
        patience: Early stopping patience (epochs without QWK improvement).

    Returns:
        float: The best QWK score achieved.
    """
    # Initialize Scaler for AMP
    scaler = torch.amp.GradScaler("cuda", enabled=Config.use_amp)

    best_qwk = -1.0
    early_stopping_counter = 0

    # Path to save the best model
    save_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training for {num_epochs} epochs on device: {device}")

    for epoch in range(num_epochs):
        # --- Training ---
        train_loss = train_fn(model, train_loader, optimizer, scheduler, device, scaler)

        # --- Validation ---
        val_loss, val_preds, val_labels = eval_fn(model, val_loader, device)

        # --- Metrics ---
        val_qwk = compute_qwk(val_labels, val_preds)

        # Print metrics (full precision)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val QWK: {val_qwk}")

        # --- Checkpointing & Early Stopping ---
        if val_qwk > best_qwk:
            print(
                f"Validation QWK improved from {best_qwk} to {val_qwk}. Saving model..."
            )
            best_qwk = val_qwk
            torch.save(model.state_dict(), save_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(
                f"No improvement in QWK. Early stopping counter: {early_stopping_counter}/{patience}"
            )

        if early_stopping_counter >= patience:
            print("Early stopping triggered.")
            break

        # Cleanup to prevent OOM
        torch.cuda.empty_cache()
        gc.collect()

    print(f"Training complete. Best QWK: {best_qwk}")
    return best_qwk
