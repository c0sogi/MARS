import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import math
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.utils import seed_everything, compute_score
from library.data_processing import get_dataloaders
from library.model import SiameseDeberta


def train_fn(model, dataloader, optimizer, scheduler, device, scaler, epoch):
    """
    Executes one training epoch with Mixed Precision and Gradient Accumulation.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Define loss function
    # CrossEntropyLoss supports soft targets (probabilities) in recent PyTorch versions
    criterion = nn.CrossEntropyLoss()

    num_batches = len(dataloader)

    for step, batch in enumerate(dataloader):
        # Move inputs to device
        input_ids_a = batch["input_ids_a"].to(device)
        attention_mask_a = batch["attention_mask_a"].to(device)
        input_ids_b = batch["input_ids_b"].to(device)
        attention_mask_b = batch["attention_mask_b"].to(device)
        scalar_features = batch["scalar_features"].to(device)
        labels = batch["labels"].to(device)

        batch_size = input_ids_a.size(0)

        # Mixed Precision Context
        with torch.amp.autocast(device_type="cuda", enabled=Config.use_fp16):
            logits = model(
                input_ids_a,
                attention_mask_a,
                input_ids_b,
                attention_mask_b,
                scalar_features,
            )
            loss = criterion(logits, labels)

            # Scale loss for gradient accumulation
            loss = loss / Config.gradient_accumulation_steps

        # Backward pass with scaler
        scaler.scale(loss).backward()

        # Update weights if accumulation steps reached
        if (step + 1) % Config.gradient_accumulation_steps == 0 or (
            step + 1
        ) == num_batches:
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()

            # Scheduler step
            if scheduler is not None:
                scheduler.step()

            # Zero gradients
            optimizer.zero_grad()

        # Track loss (multiply back by accumulation steps to get actual loss value)
        running_loss += (loss.item() * Config.gradient_accumulation_steps) * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def eval_fn(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    preds = []
    targets = []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            scalar_features = batch["scalar_features"].to(device)
            labels = batch["labels"].to(device)

            batch_size = input_ids_a.size(0)

            # Forward pass (no autocast needed for eval usually, but consistent behavior is good)
            # We stick to float32 for stability in eval unless memory is tight
            logits = model(
                input_ids_a,
                attention_mask_a,
                input_ids_b,
                attention_mask_b,
                scalar_features,
            )

            loss = criterion(logits, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply softmax to get probabilities for metric calculation
            probs = torch.softmax(logits, dim=1)
            preds.append(probs.cpu().numpy())
            targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    return epoch_loss, preds, targets


def run_training():
    """
    Main function to manage the training pipeline.
    """
    seed_everything(Config.seed)

    print(f"Initializing training on device: {Config.device}")

    # 1. Load Data
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    # 2. Initialize Model
    model = SiameseDeberta()
    model.to(Config.device)

    # 3. Setup Optimizer with Differential Learning Rates
    # Group 1: Backbone (transformers parameters)
    # Group 2: Head and Pooling (custom parameters)

    backbone_params = list(model.backbone.named_parameters())
    head_params = list(model.head.named_parameters()) + list(
        model.pooling.named_parameters()
    )

    # Filter out parameters that don't require gradients (if any)
    backbone_params = [p for n, p in backbone_params if p.requires_grad]
    head_params = [p for n, p in head_params if p.requires_grad]

    optimizer_grouped_parameters = [
        {"params": backbone_params, "lr": Config.lr_backbone},
        {"params": head_params, "lr": Config.lr_head},
    ]

    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=Config.lr_backbone,  # Default lr, though overridden by groups
        weight_decay=Config.weight_decay,
    )

    # 4. Setup Scheduler
    # Calculate total training steps
    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / Config.gradient_accumulation_steps
    )
    max_train_steps = Config.epochs * num_update_steps_per_epoch

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * max_train_steps),  # 10% warmup
        num_training_steps=max_train_steps,
    )

    # 5. Setup Scaler for Mixed Precision
    scaler = torch.amp.GradScaler(enabled=Config.use_fp16)

    # 6. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training loop...")

    for epoch in range(Config.epochs):
        start_time = time.time()

        # Train
        train_loss = train_fn(
            model, train_loader, optimizer, scheduler, Config.device, scaler, epoch
        )

        # Validate
        val_loss, val_preds, val_targets = eval_fn(model, val_loader, Config.device)

        # Calculate Log Loss metric (redundant with val_loss since it is CrossEntropy, but good for verification)
        val_log_loss = compute_score(val_targets, val_preds)

        elapsed = time.time() - start_time

        print(f"Epoch {epoch + 1}/{Config.epochs} | Time: {elapsed:.2f}s")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss:   {val_loss}")
        print(f"Log Loss:   {val_log_loss}")

        # Checkpoint & Early Stopping
        if val_loss < best_val_loss:
            print(
                f"Validation loss improved from {best_val_loss} to {val_loss}. Saving model..."
            )
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"Validation loss did not improve. Patience: {patience_counter}/{Config.patience}"
            )

        if patience_counter >= Config.patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Loss: {best_val_loss}")
