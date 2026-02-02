import torch
import torch.nn as nn
import numpy as np
import os
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from library.config import Config
from library.utils import seed_everything, compute_spearmanr
from library.dataset import get_dataloaders
from library.model import DualRoBERTa


def train_one_epoch(
    model, loader, optimizer, scheduler, criterion, device, epoch, config
):
    """
    Handles the training of one epoch.
    Manages backbone freezing/unfreezing based on the epoch.
    """
    model.train()

    # Head Warmup Strategy: Freeze backbone if needed
    freeze_backbone = epoch < config.FREEZE_BACKBONE_EPOCHS

    # We toggle requires_grad for backbone components
    backbone_modules = [model.q_backbone, model.a_backbone]

    for module in backbone_modules:
        for param in module.parameters():
            param.requires_grad = not freeze_backbone

    # The head should always be trainable
    for param in model.head.parameters():
        param.requires_grad = True

    running_loss = 0.0
    dataset_size = 0

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        # Move batch to device
        input_ids_q = batch["input_ids_q"].to(device)
        attention_mask_q = batch["attention_mask_q"].to(device)
        input_ids_a = batch["input_ids_a"].to(device)
        attention_mask_a = batch["attention_mask_a"].to(device)
        labels = batch["labels"].to(device)

        batch_size = input_ids_q.size(0)

        # Forward pass
        logits = model(input_ids_q, attention_mask_q, input_ids_a, attention_mask_a)

        # Loss calculation
        loss = criterion(logits, labels)

        # Normalize loss for gradient accumulation
        loss = loss / config.ACCUM_STEPS

        # Backward pass
        loss.backward()

        # Gradient Accumulation
        if (step + 1) % config.ACCUM_STEPS == 0 or (step + 1) == len(loader):
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        running_loss += loss.item() * config.ACCUM_STEPS * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Computes Loss and Spearman Correlation.
    """
    model.eval()

    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids_q = batch["input_ids_q"].to(device)
            attention_mask_q = batch["attention_mask_q"].to(device)
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            labels = batch["labels"].to(device)

            batch_size = input_ids_q.size(0)

            logits = model(input_ids_q, attention_mask_q, input_ids_a, attention_mask_a)
            loss = criterion(logits, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for predictions
            preds = torch.sigmoid(logits)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    score = compute_spearmanr(all_preds, all_labels)

    return epoch_loss, score


def run_training(config: Config):
    """
    Main execution function for training the model.
    """
    seed_everything(config.SEED)

    # 1. Data Loading
    loaders = get_dataloaders(config)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # 2. Model Initialization
    model = DualRoBERTa(config)
    model.to(config.device)

    # 3. Optimizer Setup (Differential Learning Rates)
    # Group parameters
    backbone_params = list(model.q_backbone.parameters()) + list(
        model.a_backbone.parameters()
    )

    head_params = list(model.head.parameters())

    optimizer_grouped_parameters = [
        {"params": backbone_params, "lr": config.BACKBONE_LR},
        {"params": head_params, "lr": config.HEAD_LR},
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, weight_decay=config.WEIGHT_DECAY, eps=config.EPS
    )

    # 4. Scheduler Setup (Phantom Scheduling)
    # Calculate total steps based on PHANTOM_EPOCHS, but we will stop early
    num_update_steps_per_epoch = len(train_loader) // config.ACCUM_STEPS
    total_training_steps = num_update_steps_per_epoch * config.PHANTOM_EPOCHS
    num_warmup_steps = int(total_training_steps * config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_score = -1.0

    print(
        f"Starting training for {config.STOP_EPOCH} epochs (Phantom Schedule: {config.PHANTOM_EPOCHS})..."
    )

    for epoch in range(config.STOP_EPOCH):
        # Train
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            config.device,
            epoch,
            config,
        )

        # Validate
        val_loss, val_score = validate(model, val_loader, criterion, config.device)

        print(
            f"Epoch {epoch+1}/{config.STOP_EPOCH} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val Spearman: {val_score}"
        )

        # Save Best Model
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"New best model saved with score: {best_score}")

    print(f"Training completed. Best Validation Score: {best_score}")
