import torch
import torch.nn as nn
from transformers import (
    AutoModelForTokenClassification,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
import os
import numpy as np
from library.config import Config


class TokenClassifier(nn.Module):
    """
    Router model based on a Transformer Encoder for token classification.
    Wraps a Hugging Face AutoModelForTokenClassification.
    """

    def __init__(
        self,
        model_name: str = Config.ROUTER_MODEL_NAME,
        num_labels: int = Config.NUM_CLASSES,
    ):
        super(TokenClassifier, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, config=self.config
        )

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss calculation.

        Returns:
            transformers.modeling_outputs.TokenClassifierOutput: Object containing loss and logits.
        """
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def save_pretrained(self, save_directory):
        """
        Saves the model and configuration to the specified directory.
        """
        self.model.save_pretrained(save_directory)
        self.config.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, load_directory):
        """
        Loads the model from a specified directory or Hugging Face hub path.
        """
        # Initialize with the path. AutoModel will handle loading weights from the dir.
        instance = cls(model_name=load_directory, num_labels=Config.NUM_CLASSES)
        return instance


def train_router(
    model,
    train_loader,
    val_loader,
    epochs=Config.ROUTER_EPOCHS,
    lr=Config.ROUTER_LR,
    device=Config.DEVICE,
    save_dir=Config.ROUTER_CHECKPOINT_DIR,
):
    """
    Training loop for the Router model.

    Args:
        model (TokenClassifier): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        device (str): Device to train on ('cuda' or 'cpu').
        save_dir (str): Directory to save the best model.

    Returns:
        TokenClassifier: The trained model (loaded with best weights).
    """
    model.to(device)

    # Optimizer configuration: Apply weight decay to all parameters except bias and LayerNorm
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": Config.ROUTER_WEIGHT_DECAY,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    # Scheduler configuration
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_acc = 0.0
    patience = 2  # Number of epochs to wait for improvement
    patience_counter = 0

    print(f"Starting Router training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        # ================= Training Phase =================
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # ================= Validation Phase =================
        model.eval()
        total_val_loss = 0.0
        correct_tokens = 0
        total_active_tokens = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
                total_val_loss += loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=2)

                # Mask out ignored tokens (-100)
                active_mask = labels != -100
                active_preds = preds[active_mask]
                active_labels = labels[active_mask]

                correct_tokens += (active_preds == active_labels).sum().item()
                total_active_tokens += active_mask.sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = (
            correct_tokens / total_active_tokens if total_active_tokens > 0 else 0.0
        )

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {avg_train_loss}")
        print(f"Val Loss: {avg_val_loss}")
        print(f"Val Accuracy: {val_acc}")

        # ================= Checkpointing & Early Stopping =================
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            print(f"New best accuracy! Saving model to {save_dir}")
            model.save_pretrained(save_dir)
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load best model before returning
    print("Loading best model weights...")
    model = TokenClassifier.from_pretrained(save_dir)
    model.to(device)

    return model
