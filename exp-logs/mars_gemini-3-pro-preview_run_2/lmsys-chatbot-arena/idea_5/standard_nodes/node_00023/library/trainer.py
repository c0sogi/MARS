import os
import time
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.utils import get_logger
from library.data import get_dataloaders
from library.model import SiameseModel, get_llrd_optimizer_params

# Initialize logger
logger = get_logger("trainer")


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        dataloader: Training dataloader.
        device: 'cuda' or 'cpu'.
        epoch: Current epoch number.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()

    total_loss = 0.0
    count = 0

    # Loss function: CrossEntropyLoss expects class indices for hard targets
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()

    for step, batch in enumerate(dataloader):
        # Move batch to device
        input_ids_a = batch["input_ids_a"].to(device)
        attention_mask_a = batch["attention_mask_a"].to(device)
        input_ids_b = batch["input_ids_b"].to(device)
        attention_mask_b = batch["attention_mask_b"].to(device)
        meta_features = batch["meta_features"].to(device)
        targets = batch["target"].to(device)

        batch_size = input_ids_a.size(0)

        # Forward pass
        # The model handles Multi-Sample Dropout internally
        logits = model(
            input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, meta_features
        )

        # Convert one-hot targets to class indices for CrossEntropyLoss
        # targets shape: [batch_size, 3] -> target_indices shape: [batch_size]
        target_indices = torch.argmax(targets, dim=1)

        loss = criterion(logits, target_indices)

        # Normalize loss for gradient accumulation (if enabled)
        if Config.GRADIENT_ACCUMULATION_STEPS > 1:
            loss = loss / Config.GRADIENT_ACCUMULATION_STEPS

        loss.backward()

        if (step + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * Config.GRADIENT_ACCUMULATION_STEPS
        count += 1

    avg_loss = total_loss / count
    elapsed = time.time() - start_time

    logger.info(f"Epoch {epoch} - Train Loss: {avg_loss:.8f} - Time: {elapsed:.2f}s")

    return avg_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: Validation dataloader.
        device: 'cuda' or 'cpu'.

    Returns:
        float: Average validation loss (Log Loss).
    """
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    count = 0

    # Store all logits and targets to compute precise metric if needed,
    # but average batch loss is sufficient for CrossEntropy

    with torch.no_grad():
        for batch in dataloader:
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            meta_features = batch["meta_features"].to(device)
            targets = batch["target"].to(device)

            logits = model(
                input_ids_a,
                attention_mask_a,
                input_ids_b,
                attention_mask_b,
                meta_features,
            )

            target_indices = torch.argmax(targets, dim=1)
            loss = criterion(logits, target_indices)

            total_loss += loss.item()
            count += 1

    avg_loss = total_loss / count
    return avg_loss


def run_fold(fold_idx, train_df, val_df, tokenizer):
    """
    Runs the training pipeline for a single fold.

    Args:
        fold_idx (int): Index of the current fold.
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        tokenizer: The tokenizer instance.

    Returns:
        float: Best validation loss achieved for this fold.
    """
    logger.info(f"Starting Fold {fold_idx}...")

    device = torch.device(Config.DEVICE)

    # 1. Prepare DataLoaders
    train_loader, val_loader = get_dataloaders(
        train_df,
        val_df,
        tokenizer,
        batch_size=Config.TRAIN_BATCH_SIZE,
        valid_batch_size=Config.VALID_BATCH_SIZE,
    )

    # 2. Initialize Model
    model = SiameseModel()
    model.to(device)

    # 3. Optimizer with LLRD
    optimizer_grouped_parameters = get_llrd_optimizer_params(model)
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=Config.LEARNING_RATE,
        eps=1e-6,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # 4. Scheduler
    num_update_steps_per_epoch = len(train_loader) // Config.GRADIENT_ACCUMULATION_STEPS
    max_train_steps = Config.EPOCHS * num_update_steps_per_epoch
    num_warmup_steps = int(max_train_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps
    )

    # 5. Training Loop
    best_val_loss = float("inf")
    best_model_path = os.path.join(
        Config.MODEL_OUTPUT_DIR, f"best_model_fold_{fold_idx}.pth"
    )

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, epoch
        )

        # Validate
        val_loss = validate(model, val_loader, device)
        logger.info(f"Epoch {epoch} - Val Loss: {val_loss:.8f}")

        # Checkpoint
        if val_loss < best_val_loss:
            logger.info(
                f"Validation loss improved from {best_val_loss:.8f} to {val_loss:.8f}. Saving model..."
            )
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
        else:
            logger.info(f"Validation loss did not improve from {best_val_loss:.8f}.")

    # Cleanup to save memory
    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    logger.info(f"Fold {fold_idx} finished. Best Val Loss: {best_val_loss:.8f}")
    return best_val_loss
