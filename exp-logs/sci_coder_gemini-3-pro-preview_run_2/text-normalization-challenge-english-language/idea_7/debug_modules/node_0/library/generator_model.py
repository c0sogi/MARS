import torch
import torch.nn as nn
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
import os
import numpy as np
from library.config import Config


class Seq2SeqNormalizer(nn.Module):
    """
    Generator model based on a Sequence-to-Sequence architecture (e.g., ByT5).
    Wraps a Hugging Face AutoModelForSeq2SeqLM.
    """

    def __init__(
        self,
        model_name: str = Config.GENERATOR_MODEL_NAME,
    ):
        super(Seq2SeqNormalizer, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, config=self.config
        )

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            labels (torch.Tensor, optional): Target token IDs for loss calculation.

        Returns:
            transformers.modeling_outputs.Seq2SeqLMOutput: Object containing loss and logits.
        """
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def generate(self, input_ids, attention_mask, max_length=Config.GEN_MAX_TARGET_LEN):
        """
        Generates sequences for input data.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            max_length (int): Maximum length of generated sequence.

        Returns:
            torch.Tensor: Generated token IDs.
        """
        return self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_length=max_length
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
        instance = cls(model_name=load_directory)
        return instance


def train_generator(
    model,
    train_loader,
    val_loader,
    epochs=Config.GEN_EPOCHS,
    lr=Config.GEN_LR,
    device=Config.DEVICE,
    save_dir=Config.GENERATOR_CHECKPOINT_DIR,
):
    """
    Training loop for the Generator model.

    Args:
        model (Seq2SeqNormalizer): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        device (str): Device to train on ('cuda' or 'cpu').
        save_dir (str): Directory to save the best model.

    Returns:
        Seq2SeqNormalizer: The trained model (loaded with best weights).
    """
    model.to(device)

    # Optimizer configuration
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": Config.GEN_WEIGHT_DECAY,
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

    best_val_loss = float("inf")
    patience = 2
    patience_counter = 0

    print(f"Starting Generator training on {device} for {epochs} epochs...")

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

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {avg_train_loss}")
        print(f"Val Loss: {avg_val_loss}")

        # ================= Checkpointing & Early Stopping =================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print(f"New best loss! Saving model to {save_dir}")
            model.save_pretrained(save_dir)
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load best model before returning
    print("Loading best model weights...")
    model = Seq2SeqNormalizer.from_pretrained(save_dir)
    model.to(device)

    return model
