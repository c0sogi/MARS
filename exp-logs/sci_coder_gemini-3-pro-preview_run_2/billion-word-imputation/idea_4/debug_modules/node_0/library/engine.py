import os
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.utils import setup_logger, set_seed
from library.modeling import GapLocatorModel, InFillerModel

# Initialize logger
logger = setup_logger("engine", os.path.join(Config.WORKING_DIR, "engine.log"))


class EarlyStopping:
    """
    Implements early stopping logic to terminate training when validation loss stops improving.
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == "min":
            self.val_score_fn = lambda x: -x
        else:
            self.val_score_fn = lambda x: x

    def __call__(self, score: float):
        if self.best_score is None:
            self.best_score = score
        elif (
            self.val_score_fn(score)
            < self.val_score_fn(self.best_score) + self.min_delta
        ):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def save_checkpoint(model: nn.Module, path: str):
    """
    Saves the model state dictionary.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Model checkpoint saved to {path}")


def evaluate_locator(
    model: nn.Module, data_loader: DataLoader, device: str
) -> Dict[str, float]:
    """
    Evaluates the Gap Locator model.
    Metrics: Average Loss, Sequence-level Accuracy (did we pick the right token index?).
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs["loss"]
            logits = outputs["logits"]

            total_loss += loss.item() * input_ids.size(0)

            # Calculate Accuracy
            # We want the index with the highest logit to match the index where label == 1
            # labels is (Batch, Seq_Len) with one 1.0 and rest 0.0, or all 0.0 (if empty sample)

            pred_indices = torch.argmax(logits, dim=1)

            # Find true indices.
            # Note: If a sample has no positive label (all zeros), argmax on labels might be 0.
            # We should check if the max value in labels is actually 1.
            true_max_vals, true_indices = torch.max(labels, dim=1)

            # Mask out samples that don't have a valid gap label (if any exist in val set)
            valid_mask = true_max_vals == 1.0

            if valid_mask.sum() > 0:
                matches = (pred_indices == true_indices) & valid_mask
                correct_predictions += matches.sum().item()
                total_samples += valid_mask.sum().item()

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

    return {"val_loss": avg_loss, "val_accuracy": accuracy}


def train_locator(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = Config.LOCATOR_EPOCHS,
    lr: float = Config.LOCATOR_LR,
    device: str = Config.DEVICE,
):
    """
    Training loop for the Stage 1 Locator model.
    """
    set_seed(Config.SEED)

    model = GapLocatorModel(model_name=Config.LOCATOR_MODEL).to(device)

    optimizer = AdamW(
        model.parameters(), lr=lr, weight_decay=Config.LOCATOR_WEIGHT_DECAY
    )

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * Config.LOCATOR_WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    early_stopper = EarlyStopping(patience=2, mode="min")
    best_val_loss = float("inf")
    save_path = os.path.join(Config.WORKING_DIR, "best_locator.pth")

    logger.info("Starting Locator Training...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs["loss"]

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * input_ids.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        # Validation
        val_metrics = evaluate_locator(model, val_loader, device)
        val_loss = val_metrics["val_loss"]
        val_acc = val_metrics["val_accuracy"]

        logger.info(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"Train Loss: {avg_train_loss}")
        logger.info(f"Val Loss: {val_loss}")
        logger.info(f"Val Accuracy: {val_acc}")

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, save_path)

        # Early Stopping
        early_stopper(val_loss)
        if early_stopper.early_stop:
            logger.info("Early stopping triggered.")
            break

    logger.info("Locator training finished.")
    return save_path


def evaluate_infiller(
    model: nn.Module, data_loader: DataLoader, device: str
) -> Dict[str, float]:
    """
    Evaluates the In-Filler model.
    Metrics: Loss, Perplexity, Accuracy (on masked tokens).
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_masked_tokens = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item() * input_ids.size(0)

            # Calculate Accuracy on masked tokens only (where label != -100)
            predictions = torch.argmax(logits, dim=-1)
            mask = labels != -100

            if mask.sum() > 0:
                correct = (predictions == labels) & mask
                correct_predictions += correct.sum().item()
                total_masked_tokens += mask.sum().item()

    avg_loss = total_loss / len(data_loader.dataset)
    perplexity = (
        np.exp(avg_loss) if avg_loss < 100 else float("inf")
    )  # Prevent overflow
    accuracy = (
        correct_predictions / total_masked_tokens if total_masked_tokens > 0 else 0.0
    )

    return {
        "val_loss": avg_loss,
        "val_perplexity": perplexity,
        "val_accuracy": accuracy,
    }


def train_infiller(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = Config.INFILLER_EPOCHS,
    lr: float = Config.INFILLER_LR,
    device: str = Config.DEVICE,
):
    """
    Training loop for the Stage 2 In-Filler model.
    """
    set_seed(Config.SEED)

    model = InFillerModel(model_name=Config.INFILLER_MODEL).to(device)

    optimizer = AdamW(
        model.parameters(), lr=lr, weight_decay=Config.INFILLER_WEIGHT_DECAY
    )

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * Config.INFILLER_WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    early_stopper = EarlyStopping(patience=2, mode="min")
    best_val_loss = float("inf")
    save_path = os.path.join(Config.WORKING_DIR, "best_infiller.pth")

    logger.info("Starting In-Filler Training...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * input_ids.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        # Validation
        val_metrics = evaluate_infiller(model, val_loader, device)
        val_loss = val_metrics["val_loss"]
        val_ppl = val_metrics["val_perplexity"]
        val_acc = val_metrics["val_accuracy"]

        logger.info(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"Train Loss: {avg_train_loss}")
        logger.info(f"Val Loss: {val_loss}")
        logger.info(f"Val Perplexity: {val_ppl}")
        logger.info(f"Val Accuracy: {val_acc}")

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, save_path)

        # Early Stopping
        early_stopper(val_loss)
        if early_stopper.early_stop:
            logger.info("Early stopping triggered.")
            break

    logger.info("In-Filler training finished.")
    return save_path
