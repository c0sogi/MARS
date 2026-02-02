import os
import torch
import logging
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup

from library.config import Config
from library.utils import seed_everything
from library.dataset import prepare_generator_data, TextNormalizationGeneratorDataset
from library.modeling import GeneratorModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """
    Performs one training epoch for the Generator model using Teacher Forcing.
    """
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward pass with labels calculates loss automatically (Teacher Forcing)
        outputs = model(input_ids, attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    return avg_loss


def validate(model, dataloader, device):
    """
    Evaluates the Generator model on the validation set.
    Returns the average Cross-Entropy Loss.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    return avg_loss


def train_generator(
    epochs=Config.GENERATOR_EPOCHS,
    batch_size=Config.GENERATOR_BATCH_SIZE,
    lr=Config.GENERATOR_LR,
    load_cached_data=True,
):
    """
    Main training loop for the Generator (Seq2Seq) model.
    Handles data loading, model initialization, training, validation, and early stopping.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    logger.info("Initializing Generator Training...")

    # 1. Prepare Data
    # The generator data preparation filters for Path B classes and constructs context windows
    train_df = prepare_generator_data("train", load_cached_data=load_cached_data)
    val_df = prepare_generator_data("val", load_cached_data=load_cached_data)

    # Handle edge case where no Path B tokens exist in the split
    if len(train_df) == 0:
        logger.warning(
            "No training data for generator found (check Path B filtering). Skipping training."
        )
        return 0.0

    tokenizer = AutoTokenizer.from_pretrained(Config.GENERATOR_MODEL_NAME)

    train_dataset = TextNormalizationGeneratorDataset(train_df, tokenizer)
    val_dataset = TextNormalizationGeneratorDataset(val_df, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 2. Initialize Model
    model = GeneratorModel(model_name=Config.GENERATOR_MODEL_NAME).to(device)

    optimizer = AdamW(
        model.parameters(), lr=lr, weight_decay=Config.GENERATOR_WEIGHT_DECAY
    )

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # 3. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    save_path = os.path.join(Config.CHECKPOINT_DIR, "generator_best")

    logger.info(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = validate(model, val_loader, device)

        # Log metrics with full precision
        logger.info(
            f"Epoch {epoch+1}: Train Loss = {train_loss}, Val Loss = {val_loss}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"New best loss! Saving model to {save_path}")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.GENERATOR_PATIENCE:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break

    return best_val_loss
