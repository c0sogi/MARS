import torch
import torch.nn as nn
import numpy as np
from transformers import get_linear_schedule_with_warmup
from library.config import TrainConfig, PathConfig
from library.model import SegmentAwareCrossEncoder
from library.utils import seed_everything, compute_spearmanr
import os


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        # Move inputs to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        q_mask = batch["q_mask"].to(device)
        a_mask = batch["a_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass with Mixed Precision
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits, _ = model(input_ids, attention_mask, q_mask, a_mask)
            loss = criterion(logits, labels)
            loss = loss / TrainConfig.grad_acc_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Optimizer step with Gradient Accumulation
        if (step + 1) % TrainConfig.grad_acc_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), TrainConfig.max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * TrainConfig.grad_acc_steps

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            q_mask = batch["q_mask"].to(device)
            a_mask = batch["a_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(
                device_type="cuda", enabled=(device.type == "cuda")
            ):
                logits, _ = model(input_ids, attention_mask, q_mask, a_mask)
                loss = criterion(logits, labels)

            total_loss += loss.item()

            # Convert logits to probabilities
            probs = torch.sigmoid(logits)

            preds_list.append(probs.cpu().numpy())
            targets_list.append(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    score = compute_spearmanr(targets, preds)

    return avg_loss, score


def run_backbone_training(train_loader, val_loader):
    """
    Main function to run the backbone training pipeline.
    """
    seed_everything(TrainConfig.seed)
    device = torch.device(TrainConfig.device)

    print(f"Initializing model on {device}...")
    model = SegmentAwareCrossEncoder()
    model.to(device)

    # Initialize Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TrainConfig.learning_rate,
        weight_decay=TrainConfig.weight_decay,
    )

    # Initialize Scheduler
    num_update_steps_per_epoch = len(train_loader) // TrainConfig.grad_acc_steps
    num_training_steps = num_update_steps_per_epoch * TrainConfig.epochs
    num_warmup_steps = int(num_training_steps * TrainConfig.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Initialize Scaler for Mixed Precision
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")

    print("Starting training...")
    for epoch in range(TrainConfig.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, epoch
        )
        val_loss, val_score = validate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{TrainConfig.epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Spearman: {val_score}")

        # Save best model based on Validation Loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(
                f"Validation loss improved. Saving model to {PathConfig.MODEL_SAVE_PATH}"
            )
            torch.save(model.state_dict(), PathConfig.MODEL_SAVE_PATH)

    print("Training complete.")
