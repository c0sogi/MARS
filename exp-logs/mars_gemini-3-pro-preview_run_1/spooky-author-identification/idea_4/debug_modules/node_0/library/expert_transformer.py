import os
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
    logging as hf_logging,
)
from library.config import Config
from library.utils import seed_everything, compute_log_loss

# Suppress HuggingFace warnings to keep output clean
hf_logging.set_verbosity_error()


class TransformerExpert:
    """
    Expert model based on a Transformer architecture (DeBERTa-v3).
    Handles training, validation, and inference for multi-class text classification.
    """

    def __init__(self, model_name=Config.DEBERTA_MODEL, num_labels=3):
        """
        Args:
            model_name (str): The HuggingFace model identifier.
            num_labels (int): Number of target classes.
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = torch.device(Config.DEVICE)
        self.model = None
        self.best_val_loss = float("inf")

        self._initialize_model()

    def _initialize_model(self):
        """
        Initializes the AutoModelForSequenceClassification and moves it to the device.
        """
        seed_everything()
        print(f"Initializing Transformer Expert: {self.model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        self.model.to(self.device)

    def fit(self, train_loader, val_loader, save_path=None):
        """
        Trains the model using the provided data loaders.
        Implements Early Stopping based on Validation Log Loss.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            save_path (str, optional): Path to save the best model.
                                       Defaults to ./working/idea_4/transformer_best.pt
        """
        seed_everything()

        if save_path is None:
            save_path = os.path.join(Config.WORKING_DIR, "transformer_best.pt")

        # Optimizer and Scheduler setup
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": Config.WEIGHT_DECAY,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=Config.LEARNING_RATE)

        # Calculate total training steps
        num_train_steps = int(len(train_loader) * Config.EPOCHS)
        num_warmup_steps = int(num_train_steps * 0.1)  # 10% warmup

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )

        loss_fn = nn.CrossEntropyLoss()

        print(f"Starting training for {Config.EPOCHS} epochs on {self.device}...")

        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            self.model.train()
            total_train_loss = 0.0

            # Training Loop
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                self.model.zero_grad()

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                loss = outputs.loss
                total_train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation Loop
            val_probs, val_true = self._predict_loop(val_loader)
            val_log_loss = compute_log_loss(val_true, val_probs)

            print(
                f"Epoch {epoch + 1}/{Config.EPOCHS} | "
                f"Train Loss: {avg_train_loss} | "
                f"Val Log Loss: {val_log_loss}"
            )

            # Early Stopping Check
            if val_log_loss < self.best_val_loss:
                self.best_val_loss = val_log_loss
                self.save(save_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        # Load best model before returning
        self.load(save_path)
        print(f"Training complete. Best Validation Log Loss: {self.best_val_loss}")
        return self

    def predict_proba(self, loader):
        """
        Generates probability predictions for the given data loader.

        Args:
            loader (DataLoader): DataLoader containing the data to predict on.

        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples, n_classes).
        """
        probs, _ = self._predict_loop(loader)
        return probs

    def _predict_loop(self, loader):
        """
        Internal helper to run inference loop.

        Args:
            loader (DataLoader): DataLoader.

        Returns:
            tuple: (probabilities, true_labels)
        """
        self.model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)

                if "labels" in batch:
                    all_labels.extend(batch["labels"].cpu().numpy())

        return np.concatenate(all_probs, axis=0), np.array(all_labels)

    def save(self, filepath):
        """
        Saves the model state dictionary.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        # print(f"Model saved to {filepath}")

    def load(self, filepath):
        """
        Loads the model state dictionary.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        # print(f"Model loaded from {filepath}")
        return self
