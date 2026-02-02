import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)
from library.configuration import Config
from library.utilities import set_seed
from library.qa_data_processing import get_qa_data, PositiveAnchoredSampler


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer and Scheduler steps
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def run_qa_training(load_cached_data=True):
    """
    Main execution function for QA training.
    Iterates over seeds, trains models, and saves the best checkpoints.
    """
    # Ensure output directory exists
    os.makedirs(Config.QA_OUTPUT_DIR, exist_ok=True)

    # 1. Load Data
    # get_qa_data handles caching logic internally
    train_ds, val_ds, _ = get_qa_data(load_cached_data=load_cached_data)

    # 2. Determine Model Initialization Path
    # Prefer TAPT weights if available
    if os.path.exists(os.path.join(Config.TAPT_OUTPUT_DIR, "config.json")):
        model_path = Config.TAPT_OUTPUT_DIR
        print(f"Initializing QA model from TAPT weights: {model_path}")
    else:
        model_path = Config.MODEL_CHECKPOINT
        print(f"Initializing QA model from base weights: {model_path}")

    # 3. Training Loop (Ensembling over seeds)
    for seed in Config.SEEDS:
        print(f"\n--- Starting training for Seed {seed} ---")
        set_seed(seed)

        # Initialize Model
        # 3 labels: O (0), B-ANS (1), I-ANS (2)
        model = AutoModelForTokenClassification.from_pretrained(
            model_path, num_labels=3
        )
        model.to(Config.DEVICE)

        # Prepare DataLoaders
        # Use PositiveAnchoredSampler for training to ensure positive samples in every batch
        train_sampler = PositiveAnchoredSampler(
            train_ds, batch_size=Config.TRAIN_BATCH_SIZE
        )

        # Note: batch_size in DataLoader must match sampler's batch_size logic
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.TRAIN_BATCH_SIZE,
            sampler=train_sampler,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=Config.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        total_steps = len(train_loader) * Config.EPOCHS
        warmup_steps = int(total_steps * Config.WARMUP_RATIO)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # Training State
        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        early_stopping_patience = 3

        print(
            f"Training for {Config.EPOCHS} epochs with patience {early_stopping_patience}..."
        )

        for epoch in range(Config.EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, Config.DEVICE, epoch
            )
            val_loss = validate(model, val_loader, Config.DEVICE)

            # Print full precision metrics
            print(
                f"Epoch {epoch + 1} | Train Loss: {train_loss} | Val Loss: {val_loss}"
            )

            # Checkpointing and Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Save best model for this seed
        save_path = os.path.join(Config.QA_OUTPUT_DIR, f"model_seed_{seed}.pt")
        print(
            f"Saving best model for seed {seed} to {save_path} (Best Val Loss: {best_val_loss})"
        )

        if best_model_state is not None:
            torch.save(best_model_state, save_path)
        else:
            # Fallback if training failed to improve (unlikely)
            torch.save(model.state_dict(), save_path)
