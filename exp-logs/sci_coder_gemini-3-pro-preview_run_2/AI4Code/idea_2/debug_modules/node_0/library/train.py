import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from library.config import (
    MODEL_NAME,
    WORKING_DIR,
    BATCH_SIZE,
    VAL_BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_WORKERS,
    ACCUMULATE_GRAD_BATCHES,
    PATIENCE,
    MIN_DELTA,
    DEVICE,
    SEED,
    MAX_LEN,
)
from library.utils import seed_everything
from library.preprocess import create_training_dataframe
from library.dataset import MarkdownRankDataset
from library.model import ContextAwareRanker


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs["loss"]

        # Normalize loss for gradient accumulation
        loss = loss / ACCUMULATE_GRAD_BATCHES
        loss.backward()

        total_loss += loss.item() * ACCUMULATE_GRAD_BATCHES

        # Optimizer step
        if (step + 1) % ACCUMULATE_GRAD_BATCHES == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch + 1} Training Loss: {avg_loss}")
    return avg_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs["loss"]
            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Validation Loss: {avg_loss}")
    return avg_loss


def train_model(load_cached_data=True, debug=False):
    """
    Main function to train the ContextAwareRanker model.

    Args:
        load_cached_data (bool): Whether to load pre-processed dataframes from cache.
        debug (bool): Whether to run in debug mode with a smaller dataset.
    """
    # Set reproducibility
    seed_everything(SEED)

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)
    model_save_path = os.path.join(WORKING_DIR, "best_model.pth")

    print(f"Initializing model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = ContextAwareRanker(model_name=MODEL_NAME).to(DEVICE)

    # Load Data
    print("Loading data...")
    df_train, df_val = create_training_dataframe(
        load_cached_data=load_cached_data, debug=debug
    )

    train_dataset = MarkdownRankDataset(df_train, tokenizer, max_len=MAX_LEN)
    val_dataset = MarkdownRankDataset(df_val, tokenizer, max_len=MAX_LEN)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    num_training_steps = (len(train_loader) // ACCUMULATE_GRAD_BATCHES) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    # Training Loop with Early Stopping
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, DEVICE, epoch
        )
        val_loss = validate(model, val_loader, DEVICE)

        # Check for improvement
        if val_loss < (best_val_loss - MIN_DELTA):
            print(
                f"Validation loss improved from {best_val_loss} to {val_loss}. Saving model..."
            )
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            print(
                f"No improvement in validation loss. Patience: {patience_counter}/{PATIENCE}"
            )

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Loss: {best_val_loss}")
    print(f"Best model saved to {model_save_path}")
