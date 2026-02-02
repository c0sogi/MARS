import os
import time
import numpy as np
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.utils import compute_pearson, get_optimizer_grouped_parameters
from library.model import CustomModel


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        dataloader: The training dataloader.
        device: The device to train on.
        epoch: The current epoch number.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()

    dataset_size = 0
    running_loss = 0.0

    for step, data in enumerate(dataloader):
        input_ids = data["input_ids"].to(device, dtype=torch.long)
        attention_mask = data["attention_mask"].to(device, dtype=torch.long)
        targets = data["labels"].to(device, dtype=torch.float)

        batch_size = input_ids.size(0)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        # Output shape is (batch_size, 1), targets is (batch_size)
        outputs = outputs.view(-1)

        loss = nn.MSELoss()(outputs, targets)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def valid_one_epoch(model, dataloader, device):
    """
    Validates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: The validation dataloader.
        device: The device to validate on.

    Returns:
        tuple: (Average validation loss, Pearson correlation score)
    """
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(device, dtype=torch.long)
            targets = data["labels"].to(device, dtype=torch.float)

            batch_size = input_ids.size(0)

            outputs = model(input_ids, attention_mask)
            outputs = outputs.view(-1)

            loss = nn.MSELoss()(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    pearson_score = compute_pearson(all_preds, all_labels)

    return epoch_loss, pearson_score


def fit(train_loader, val_loader):
    """
    Orchestrates the training process: initialization, looping through epochs,
    validation, early stopping, and saving the best model.

    Args:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
    """
    # Instantiate Config to ensure directories are created
    _ = Config()

    device = Config.device

    # Initialize Model
    model = CustomModel()
    model.to(device)

    # Optimizer with LLRD
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model,
        learning_rate=Config.learning_rate,
        weight_decay=Config.weight_decay,
        llrd_decay=Config.llrd_decay,
    )

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=Config.learning_rate, eps=Config.eps
    )

    # Scheduler
    num_train_steps = len(train_loader) * Config.epochs
    num_warmup_steps = int(num_train_steps * Config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    print(f"Training started on {device}")

    best_pearson = -float("inf")
    patience = 2  # Early stopping patience
    patience_counter = 0

    for epoch in range(Config.epochs):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, epoch
        )
        val_loss, val_pearson = valid_one_epoch(model, val_loader, device)

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"Epoch {epoch + 1}/{Config.epochs} | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Pearson: {val_pearson}")

        # Save Best Model
        if val_pearson > best_pearson:
            print(
                f"Validation Pearson improved from {best_pearson} to {val_pearson}. Saving model..."
            )
            best_pearson = val_pearson
            torch.save(model.state_dict(), Config.model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Pearson: {best_pearson}")
