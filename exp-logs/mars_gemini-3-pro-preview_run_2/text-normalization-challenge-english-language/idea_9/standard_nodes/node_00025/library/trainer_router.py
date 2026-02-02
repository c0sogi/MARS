import os
import torch
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from library.config import Config
from library.utils import seed_everything
from library.dataset import prepare_router_data, TextNormalizationRouterDataset
from library.modeling import RouterModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """
    Performs one training epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
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
    Evaluates the model on the validation set and returns the accuracy.
    Ignores tokens with label -100 (subwords/padding).
    """
    model.eval()
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=2)

            # Filter out -100 labels (padding/special tokens)
            active_loss = labels.view(-1) != -100
            active_logits = preds.view(-1)[active_loss]
            active_labels = labels.view(-1)[active_loss]

            val_preds.extend(active_logits.cpu().numpy())
            val_labels.extend(active_labels.cpu().numpy())

    if len(val_labels) == 0:
        return 0.0

    # Calculate exact match accuracy
    accuracy = np.mean(np.array(val_preds) == np.array(val_labels))
    return accuracy


def train_router(
    epochs=Config.ROUTER_EPOCHS,
    batch_size=Config.ROUTER_BATCH_SIZE,
    lr=Config.ROUTER_LR,
    load_cached_data=True,
):
    """
    Main training loop for the Router model.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    logger.info("Initializing Router Training...")

    # 1. Load and Prepare Data
    train_df = prepare_router_data("train", load_cached_data=load_cached_data)
    val_df = prepare_router_data("val", load_cached_data=load_cached_data)

    tokenizer = AutoTokenizer.from_pretrained(Config.ROUTER_MODEL_NAME)

    train_dataset = TextNormalizationRouterDataset(train_df, tokenizer, is_test=False)
    val_dataset = TextNormalizationRouterDataset(val_df, tokenizer, is_test=False)

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

    # 2. Initialize Model, Optimizer, Scheduler
    model = RouterModel(
        model_name=Config.ROUTER_MODEL_NAME, num_labels=Config.NUM_LABELS
    ).to(device)

    optimizer = AdamW(
        model.parameters(), lr=lr, weight_decay=Config.ROUTER_WEIGHT_DECAY
    )

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # 3. Training Loop with Early Stopping
    best_val_accuracy = -1.0
    patience_counter = 0
    save_path = os.path.join(Config.CHECKPOINT_DIR, "router_best")

    logger.info(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_accuracy = validate(model, val_loader, device)

        # Print metrics with full precision
        logger.info(
            f"Epoch {epoch+1}: Train Loss = {train_loss}, Val Accuracy = {val_accuracy}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            logger.info(f"New best accuracy! Saving model to {save_path}")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.ROUTER_PATIENCE:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break

    return best_val_accuracy
