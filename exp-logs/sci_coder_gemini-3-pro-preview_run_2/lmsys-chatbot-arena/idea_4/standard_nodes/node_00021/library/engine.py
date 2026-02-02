import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import log_loss
from transformers import get_linear_schedule_with_warmup
from library.config import Config
from library.utils import seed_everything
from library.model import SiameseDeberta


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, scaler):
    """
    Trains the model for one epoch using Mixed Precision (FP16).
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        # Move inputs to device
        input_ids_a = batch["input_ids_a"].to(device)
        attention_mask_a = batch["attention_mask_a"].to(device)
        input_ids_b = batch["input_ids_b"].to(device)
        attention_mask_b = batch["attention_mask_b"].to(device)
        meta_features = batch["meta_features"].to(device)
        targets = batch["labels"].to(device)

        batch_size = input_ids_a.size(0)

        # Convert one-hot targets to class indices for CrossEntropyLoss
        # targets shape: (batch, 3) -> target_indices shape: (batch,)
        target_indices = torch.argmax(targets, dim=1)

        optimizer.zero_grad()

        # Mixed Precision Forward Pass
        with torch.cuda.amp.autocast(enabled=Config.USE_FP16):
            logits = model(
                input_ids_a,
                attention_mask_a,
                input_ids_b,
                attention_mask_b,
                meta_features,
            )
            loss = criterion(logits, target_indices)

        # Mixed Precision Backward Pass
        scaler.scale(loss).backward()

        # Gradient Clipping (Unscale first)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and Log Loss metric.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            meta_features = batch["meta_features"].to(device)
            targets = batch["labels"].to(device)

            batch_size = input_ids_a.size(0)
            target_indices = torch.argmax(targets, dim=1)

            # Forward pass (no autocast needed for eval usually, but consistent dtype helps)
            with torch.cuda.amp.autocast(enabled=Config.USE_FP16):
                logits = model(
                    input_ids_a,
                    attention_mask_a,
                    input_ids_b,
                    attention_mask_b,
                    meta_features,
                )
                loss = criterion(logits, target_indices)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Log Loss
    # eps='auto' is default in sklearn, but explicit clipping is handled by library usually.
    # We use the raw probabilities.
    metric_score = log_loss(all_targets, all_preds)

    return epoch_loss, metric_score


def train_model(train_loader, val_loader):
    """
    Main function to train the model with Early Stopping.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Initialize Model
    model = SiameseDeberta()
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    num_training_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=Config.USE_FP16)

    # Training Loop Variables
    best_log_loss = float("inf")
    patience = 0
    early_stopping_patience = (
        1  # Stop if no improvement after 1 epoch given strict time/resource limits
    )

    print(f"Starting training on device: {device}")

    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, scaler
        )
        print(f"Train Loss: {train_loss}")

        # Validate
        val_loss, val_log_loss = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss}")
        print(f"Val Log Loss: {val_log_loss}")

        # Checkpoint & Early Stopping
        if val_log_loss < best_log_loss:
            print(
                f"Validation Log Loss improved from {best_log_loss} to {val_log_loss}. Saving model..."
            )
            best_log_loss = val_log_loss
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience = 0
        else:
            print(f"Validation Log Loss did not improve.")
            patience += 1
            if patience >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Log Loss: {best_log_loss}")
    return best_log_loss


def infer(test_loader):
    """
    Loads the best model and generates predictions for the test set.
    Returns:
        ids (numpy array): Array of IDs.
        predictions (numpy array): Array of shape (N, 3) with probabilities.
    """
    device = Config.DEVICE

    # Load Model Architecture
    model = SiameseDeberta()
    model.to(device)

    # Load Weights
    if not torch.cuda.is_available():
        state_dict = torch.load(
            Config.MODEL_SAVE_PATH, map_location=torch.device("cpu")
        )
    else:
        state_dict = torch.load(Config.MODEL_SAVE_PATH)

    model.load_state_dict(state_dict)
    model.eval()

    all_ids = []
    all_preds = []

    print("Starting inference...")

    with torch.no_grad():
        for batch in test_loader:
            ids = batch["ids"]
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            meta_features = batch["meta_features"].to(device)

            # Disable autocast to prevent overflow during inference (Cite debug_lesson_2)
            logits = model(
                input_ids_a,
                attention_mask_a,
                input_ids_b,
                attention_mask_b,
                meta_features,
            )

            probs = torch.softmax(logits, dim=1)

            all_ids.extend(ids.numpy())
            all_preds.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_ids = np.array(all_ids)

    return all_ids, all_preds
