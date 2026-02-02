import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup, AutoTokenizer

from library.config import Config
from library.utils import get_device, save_checkpoint, load_checkpoint
from library.dataset import get_dataloader, prepare_data, ChatbotDataset, CollateFn
from library.model import SiameseDeberta


def train_fn(dataloader, model, optimizer, scheduler, device, config, epoch):
    """
    Executes one training epoch.
    """
    model.train()
    scaler = GradScaler(enabled=config.fp16)

    total_loss = 0.0
    count = 0

    # Loss function: CrossEntropyLoss supports soft targets (probabilities)
    criterion = nn.CrossEntropyLoss()

    # Accumulation steps
    accum_steps = config.gradient_accumulation_steps

    for step, batch in enumerate(dataloader):
        # Move batch to device
        # Note: The model's forward method handles moving tensor fields to device,
        # but we need targets on device for loss calculation.
        if "target" in batch:
            targets = batch["target"].to(device)
        else:
            # Should not happen in training
            continue

        # Forward pass with Mixed Precision
        with autocast(enabled=config.fp16):
            logits = model(batch)
            loss = criterion(logits, targets)

            if accum_steps > 1:
                loss = loss / accum_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Update weights
        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        # Logging
        loss_val = loss.item() * accum_steps
        total_loss += loss_val
        count += 1

    avg_loss = total_loss / count
    print(f"Epoch {epoch} Training Loss: {avg_loss}")
    return avg_loss


def eval_fn(dataloader, model, device, config):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0.0
    count = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            if "target" in batch:
                targets = batch["target"].to(device)

            # Forward pass (no autocast needed for eval usually, but good for consistency/speed)
            with autocast(enabled=config.fp16):
                logits = model(batch)
                loss = criterion(logits, targets)

            total_loss += loss.item()
            count += 1

    avg_loss = total_loss / count
    print(f"Validation Loss: {avg_loss}")
    return avg_loss


def predict_fn(dataloader, model, device, config):
    """
    Generates soft probability predictions for a dataloader.
    Returns: numpy array of shape (N, 3)
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in dataloader:
            with autocast(enabled=config.fp16):
                logits = model(batch)

            probs = torch.softmax(logits, dim=1)
            preds.append(probs.cpu().numpy())

    return np.concatenate(preds, axis=0)


def run_training(config: Config):
    """
    Orchestrates the entire training loop with early stopping.
    """
    device = get_device()
    print(f"Using device: {device}")

    # 1. Data Loaders
    train_loader = get_dataloader(config, partition="train")
    val_loader = get_dataloader(config, partition="val")

    # 2. Model
    model = SiameseDeberta(config)
    model.to(device)

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Calculate total training steps
    num_update_steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
    max_train_steps = config.epochs * num_update_steps_per_epoch
    num_warmup_steps = int(config.warmup_ratio * max_train_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps
    )

    # 4. Training Loop with Early Stopping
    best_score = float("inf")
    patience_counter = 0
    best_model_path = config.get_model_save_path()

    for epoch in range(1, config.epochs + 1):
        print(f"\nStarting Epoch {epoch}/{config.epochs}")

        # Train
        train_loss = train_fn(
            train_loader, model, optimizer, scheduler, device, config, epoch
        )

        # Validate
        val_loss = eval_fn(val_loader, model, device, config)

        print(f"Epoch {epoch} - Train Loss: {train_loss} | Val Loss: {val_loss}")

        # Checkpoint & Early Stopping
        if val_loss < best_score:
            print(
                f"Validation Score Improved ({best_score} -> {val_loss}). Saving model..."
            )
            best_score = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_score, best_model_path
            )
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{config.patience}")

        if patience_counter >= config.patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Loss: {best_score}")

    # Load best model for return
    load_checkpoint(best_model_path, model, device=device)
    return model


def generate_submission(model, config: Config):
    """
    Generates predictions for the test set, handles TTA, and saves submission.csv.
    """
    device = get_device()
    model.to(device)
    model.eval()

    print("\nGenerating Submission...")

    # 1. Load Test Data
    # We use prepare_data to get the raw dataframe
    test_df = prepare_data(config, partition="test")

    # Setup Tokenizer and CollateFn for manual dataloader creation
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    collate_fn = CollateFn(tokenizer)

    # --- Prediction 1: Original (A, B) ---
    print("Predicting on original test set...")
    ds_original = ChatbotDataset(test_df, tokenizer, config.max_length, mode="test")
    loader_original = torch.utils.data.DataLoader(
        ds_original,
        batch_size=config.valid_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    preds_original = predict_fn(loader_original, model, device, config)

    final_preds = preds_original

    # --- Prediction 2: TTA (B, A) ---
    if config.use_tta:
        print("Predicting on swapped test set (TTA)...")

        # Create swapped dataframe
        df_swapped = test_df.copy()
        df_swapped.rename(
            columns={"response_a": "response_b_temp", "response_b": "response_a_temp"},
            inplace=True,
        )
        df_swapped.rename(
            columns={"response_b_temp": "response_b", "response_a_temp": "response_a"},
            inplace=True,
        )

        ds_swapped = ChatbotDataset(
            df_swapped, tokenizer, config.max_length, mode="test"
        )
        loader_swapped = torch.utils.data.DataLoader(
            ds_swapped,
            batch_size=config.valid_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        preds_swapped = predict_fn(loader_swapped, model, device, config)

        # Invert predictions to match original order
        # Original: [A wins, B wins, Tie]
        # Swapped Input (B, A) -> Output: [First(B) wins, Second(A) wins, Tie]
        # So Swapped Output col 0 is B wins, col 1 is A wins.
        # We need to map back to [A wins, B wins, Tie]
        # New A = Swapped B (col 0)
        # New B = Swapped A (col 1)
        # New Tie = Swapped Tie (col 2)

        preds_swapped_aligned = np.zeros_like(preds_swapped)
        preds_swapped_aligned[:, 0] = preds_swapped[:, 1]  # A wins
        preds_swapped_aligned[:, 1] = preds_swapped[:, 0]  # B wins
        preds_swapped_aligned[:, 2] = preds_swapped[:, 2]  # Tie

        # Average
        final_preds = (preds_original + preds_swapped_aligned) / 2.0

    # 3. Create Submission DataFrame
    sub_df = pd.DataFrame(
        {
            "id": test_df["id"],
            "winner_model_a": final_preds[:, 0],
            "winner_model_b": final_preds[:, 1],
            "winner_tie": final_preds[:, 2],
        }
    )

    # 4. Save
    os.makedirs(config.submission_dir, exist_ok=True)
    sub_df.to_csv(config.submission_path, index=False)
    print(f"Submission saved to {config.submission_path}")
    print(sub_df.head())
