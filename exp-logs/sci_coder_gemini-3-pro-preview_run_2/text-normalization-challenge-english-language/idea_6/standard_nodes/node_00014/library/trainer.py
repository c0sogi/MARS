import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any

from library.config import Config
from library.utils import get_logger, get_device, ensure_dir
from library.model import TransformerTokenClassifier
from library.label_manager import LabelEngineer
from library.transformations import TransformationRegistry

logger = get_logger("trainer")


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to handle variable-length lists in the dataset.
    Stacks tensors and preserves lists for metadata.
    """
    # Stack tensors
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    # Preserve lists (list of lists)
    word_ids = [item["word_ids"] for item in batch]
    raw_tokens = [item["raw_tokens"] for item in batch]
    submission_ids = [item["submission_ids"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "word_ids": word_ids,
        "raw_tokens": raw_tokens,
        "submission_ids": submission_ids,
    }


class Trainer:
    """
    Trainer class for the TransformerTokenClassifier.
    Handles training loop, validation, early stopping, and checkpointing.
    """

    def __init__(
        self,
        model: TransformerTokenClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        device: torch.device,
        epochs: int = Config.EPOCHS,
        patience: int = Config.EARLY_STOPPING_PATIENCE,
        max_grad_norm: float = Config.MAX_GRAD_NORM,
        save_dir: str = Config.MODEL_CHECKPOINT_DIR,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.max_grad_norm = max_grad_norm
        self.save_dir = save_dir

        self.model.to(self.device)
        # Ignore index -100 is standard for masked tokens in HuggingFace/PyTorch
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def train(self):
        logger.info(f"Starting training on device: {self.device}")
        best_val_acc = -1.0
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            for batch_idx, batch in enumerate(self.train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(input_ids, attention_mask, labels=labels)
                loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()

                if (batch_idx + 1) % 100 == 0:
                    logger.info(
                        f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(self.train_loader)} | Loss: {loss.item():.6f}"
                    )

            avg_train_loss = total_loss / len(self.train_loader)
            logger.info(f"Epoch {epoch+1} Average Train Loss: {avg_train_loss:.6f}")

            # Validation
            val_acc, val_loss = self.evaluate()
            logger.info(
                f"Epoch {epoch+1} Validation Loss: {val_loss:.6f} | Accuracy: {val_acc:.6f}"
            )

            # Early Stopping
            if val_acc > best_val_acc:
                logger.info(
                    f"Validation accuracy improved ({best_val_acc:.6f} -> {val_acc:.6f}). Saving model..."
                )
                best_val_acc = val_acc
                self.model.save_pretrained(self.save_dir)
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(
                    f"No improvement. Patience: {patience_counter}/{self.patience}"
                )

            if patience_counter >= self.patience:
                logger.info("Early stopping triggered.")
                break

        logger.info(f"Training complete. Best Validation Accuracy: {best_val_acc:.6f}")

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids, attention_mask, labels=labels)
                total_loss += outputs.loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)

                # Mask ignored tokens
                active_mask = labels != -100
                active_preds = preds[active_mask]
                active_labels = labels[active_mask]

                all_preds.extend(active_preds.cpu().numpy())
                all_labels.extend(active_labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0

        return accuracy, avg_loss


def run_training_pipeline(
    train_dataset,
    val_dataset,
    epochs: int = Config.EPOCHS,
    batch_size: int = Config.TRAIN_BATCH_SIZE,
    val_batch_size: int = Config.VAL_BATCH_SIZE,
    learning_rate: float = Config.LEARNING_RATE,
    weight_decay: float = Config.WEIGHT_DECAY,
    warmup_ratio: float = Config.WARMUP_RATIO,
):
    device = get_device()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )

    model = TransformerTokenClassifier(pretrained_model_name=Config.MODEL_NAME)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs,
    )

    trainer.train()
    return trainer


def generate_submission(
    test_dataset,
    output_path: str = Config.SUBMISSION_PATH,
    batch_size: int = Config.VAL_BATCH_SIZE,
):
    logger.info("Generating submission file...")
    device = get_device()

    # Load best model
    try:
        model = TransformerTokenClassifier.from_pretrained(Config.MODEL_CHECKPOINT_DIR)
    except Exception as e:
        logger.error(f"Failed to load model from {Config.MODEL_CHECKPOINT_DIR}: {e}")
        return

    model.to(device)
    model.eval()

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )

    # Setup transformation logic
    label_engineer = LabelEngineer()
    label_engineer._load_or_create_label_encoder()
    id_to_name = {i: name for i, name in enumerate(label_engineer.label_names)}
    registry = TransformationRegistry()

    results = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()  # (B, L)

            # Iterate through batch
            for i in range(len(batch["submission_ids"])):
                # Get sample data
                sample_preds = preds[i]
                word_ids = batch["word_ids"][i]
                raw_tokens = batch["raw_tokens"][i]
                submission_ids = batch["submission_ids"][i]

                # Map predictions to tokens
                previous_word_idx = None
                processed_indices = set()

                for seq_idx, word_idx in enumerate(word_ids):
                    if word_idx == -1:
                        continue

                    if word_idx != previous_word_idx:
                        # Start of a new word token
                        if word_idx < len(raw_tokens):
                            raw_token = raw_tokens[word_idx]
                            sub_id = submission_ids[word_idx]
                            pred_label_id = sample_preds[seq_idx]

                            # Transform
                            label_name = id_to_name.get(pred_label_id, "TRANS_PLAIN")
                            normalized_text = registry.apply(label_name, raw_token)

                            results.append({"id": sub_id, "after": normalized_text})
                            processed_indices.add(word_idx)

                    previous_word_idx = word_idx

                # Handle truncated tokens (fallback to identity/plain)
                # This ensures we have a prediction for every input token, even if truncated
                for idx in range(len(raw_tokens)):
                    if idx not in processed_indices:
                        results.append(
                            {
                                "id": submission_ids[idx],
                                "after": raw_tokens[idx],  # Default to raw text
                            }
                        )

    # Create DataFrame
    df_sub = pd.DataFrame(results)

    # Ensure output directory exists
    ensure_dir(output_path)

    # Save
    df_sub.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path} with {len(df_sub)} rows.")
