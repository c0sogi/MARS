import os
import gc
import time
import numpy as np
import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup

from library.config import Config
from library.utils import calculate_log_loss
from library.modeling import DebertaClassifier, get_llrd_optimizer_params


def train_fn(model, dataloader, optimizer, scheduler, device, epoch):
    """
    Performs one epoch of training.
    """
    model.train()

    scaler = torch.amp.GradScaler("cuda")
    train_loss = 0.0
    dataset_size = 0
    running_loss = 0.0

    # Gradient accumulation steps
    accum_steps = Config.GRAD_ACCUM_STEPS

    optimizer.zero_grad()

    for step, data in enumerate(dataloader):
        ids = data["input_ids"].to(device, dtype=torch.long)
        mask = data["attention_mask"].to(device, dtype=torch.long)
        targets = data["labels"].to(device, dtype=torch.long)

        batch_size = ids.size(0)

        with torch.amp.autocast("cuda"):
            outputs = model(ids, mask)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * accum_steps) * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def eval_fn(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Returns the average loss and the predictions (probabilities).
    """
    model.eval()

    preds = []
    final_targets = []

    with torch.no_grad():
        for data in dataloader:
            ids = data["input_ids"].to(device, dtype=torch.long)
            mask = data["attention_mask"].to(device, dtype=torch.long)
            targets = data["labels"].to(device, dtype=torch.long)

            with torch.amp.autocast("cuda"):
                outputs = model(ids, mask)

            # Convert logits to probabilities
            probs = torch.softmax(outputs, dim=1)

            preds.append(probs.cpu().numpy())
            final_targets.append(targets.cpu().numpy())

    preds = np.concatenate(preds)
    final_targets = np.concatenate(final_targets)

    # Calculate metric
    loss = calculate_log_loss(final_targets, preds)

    return loss, preds


def run_transformer_fold(fold, train_loader, val_loader):
    """
    Trains and validates the Transformer model for a single fold.

    Args:
        fold (int): The current fold index.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.

    Returns:
        tuple: (best_val_loss, best_preds)
    """
    print(f"\n{'='*20} Fold {fold} {'='*20}")

    device = torch.device(Config.DEVICE)

    # Initialize Model
    model = DebertaClassifier(Config.MODEL_NAME, num_classes=3)
    model.to(device)

    # Optimizer with LLRD
    optimizer_parameters = get_llrd_optimizer_params(
        model,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        decay_factor=Config.LLRD_DECAY,
    )
    optimizer = torch.optim.AdamW(optimizer_parameters, eps=Config.EPS)

    # Scheduler
    # Calculate total training steps
    num_update_steps_per_epoch = len(train_loader) // Config.GRAD_ACCUM_STEPS
    max_train_steps = Config.EPOCHS * num_update_steps_per_epoch
    num_warmup_steps = int(max_train_steps * Config.NUM_WARMUP_STEPS_RATIO)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps
    )

    # Training Loop
    best_loss = float("inf")
    best_preds = None
    patience_counter = 0

    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"expert_a_fold_{fold}.pt")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_fn(model, train_loader, optimizer, scheduler, device, epoch)
        val_loss, val_preds = eval_fn(model, val_loader, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss}"
        )

        # Checkpointing & Early Stopping
        if val_loss < best_loss:
            print(
                f"Validation loss improved ({best_loss} -> {val_loss}). Saving model..."
            )
            best_loss = val_loss
            best_preds = val_preds
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            print(
                f"Validation loss did not improve. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # Cleanup
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    return best_loss, best_preds
