import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import log_loss, accuracy_score

from library.config import Config
from library.utils import seed_everything, clip_probabilities
from library.data_processing import TransformerDataset


class TransformerExpert(nn.Module):
    """
    Contextual Expert model wrapping a Transformer backbone (DeBERTa)
    with a custom classification head.
    """

    def __init__(self, model_name=Config.MODEL_NAME, num_classes=3):
        super(TransformerExpert, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        # Classification head
        # DeBERTa v3 uses the first token (CLS) for classification tasks
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)

        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask, labels=None):
        # Pass through backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Extract CLS token state (batch_size, hidden_size)
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]

        # Pass through classifier
        logits = self.classifier(cls_embedding)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return logits, loss


class Trainer:
    """
    Helper class to manage training, validation, and prediction for the TransformerExpert.
    """

    def __init__(self, device=Config.DEVICE):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.model = TransformerExpert(
            Config.MODEL_NAME, num_classes=len(Config.LABELS)
        )
        self.model.to(self.device)

    def fit(self, train_texts, train_labels, val_texts, val_labels, fold_idx=0):
        """
        Trains the model using the provided data, implementing early stopping.

        Args:
            train_texts: List/Series of training text.
            train_labels: List/Series of training labels (strings).
            val_texts: List/Series of validation text.
            val_labels: List/Series of validation labels (strings).
            fold_idx: Integer index for the current fold (used for file naming).

        Returns:
            float: Best validation log loss achieved.
        """
        seed_everything(Config.SEED)

        # Prepare Datasets
        train_dataset = TransformerDataset(
            texts=train_texts,
            labels=train_labels,
            tokenizer=self.tokenizer,
            max_length=Config.MAX_LENGTH,
        )
        val_dataset = TransformerDataset(
            texts=val_texts,
            labels=val_labels,
            tokenizer=self.tokenizer,
            max_length=Config.MAX_LENGTH,
        )

        # Prepare DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Optimizer & Scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        num_train_steps = len(train_loader) * Config.EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * num_train_steps),
            num_training_steps=num_train_steps,
        )

        # Training State
        best_loss = float("inf")
        best_model_wts = copy.deepcopy(self.model.state_dict())
        patience_counter = 0

        print(f"Starting training for fold {fold_idx}...")

        for epoch in range(Config.EPOCHS):
            # --- Training ---
            self.model.train()
            train_loss_accum = 0.0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()

                logits, loss = self.model(input_ids, attention_mask, labels)

                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss_accum += loss.item() * input_ids.size(0)

            avg_train_loss = train_loss_accum / len(train_dataset)

            # --- Validation ---
            val_loss, val_acc = self.validate(val_loader)

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss} | Val Acc: {val_acc}"
            )

            # --- Early Stopping ---
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                self.save_model(fold_idx)
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Reload best model for future use
        self.model.load_state_dict(best_model_wts)
        print(f"Fold {fold_idx} finished. Best Val Loss: {best_loss}")
        return best_loss

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        val_loss_accum = 0.0
        all_preds = []
        all_labels = []

        # Use sum reduction to aggregate correctly manually
        loss_fct = nn.CrossEntropyLoss(reduction="sum")

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits, _ = self.model(input_ids, attention_mask)

                loss = loss_fct(logits, labels)
                val_loss_accum += loss.item()

                probs = torch.softmax(logits, dim=1)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # Compute average loss
        avg_val_loss = val_loss_accum / len(val_loader.dataset)

        # Concatenate predictions
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Calculate Metrics
        # Clip probabilities for stability in log_loss calculation
        clipped_preds = clip_probabilities(all_preds)

        # Provide labels explicitly to handle cases where a batch might miss a class
        labels_idx = list(range(len(Config.LABELS)))
        metric_loss = log_loss(all_labels, clipped_preds, labels=labels_idx)

        pred_classes = np.argmax(all_preds, axis=1)
        acc = accuracy_score(all_labels, pred_classes)

        return metric_loss, acc

    def predict(self, texts):
        """
        Generates probability predictions for the given list of texts.

        Args:
            texts: List/Series of text strings.

        Returns:
            np.ndarray: Array of shape (n_samples, n_classes) with probabilities.
        """
        self.model.eval()

        dataset = TransformerDataset(
            texts=texts,
            labels=None,
            tokenizer=self.tokenizer,
            max_length=Config.MAX_LENGTH,
        )

        loader = DataLoader(
            dataset,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        all_probs = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                logits, _ = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    def save_model(self, fold_idx):
        """Saves the model state dict to the working directory."""
        output_path = os.path.join(
            Config.WORKING_DIR, f"transformer_fold_{fold_idx}.pt"
        )
        torch.save(self.model.state_dict(), output_path)

    def load_model(self, fold_idx):
        """Loads the model state dict from the working directory."""
        path = os.path.join(Config.WORKING_DIR, f"transformer_fold_{fold_idx}.pt")
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Loaded model from {path}")
        else:
            print(f"Model file not found: {path}")
