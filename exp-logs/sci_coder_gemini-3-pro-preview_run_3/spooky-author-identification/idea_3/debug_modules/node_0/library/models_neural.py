import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from library.config import Config
from library.utils import calculate_log_loss, set_seed


class TransformerDataset(Dataset):
    """
    PyTorch Dataset for Transformer models.
    Handles tokenization and input formatting.
    """

    def __init__(self, texts, targets=None, tokenizer=None, max_len=Config.MAX_LEN):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if self.targets is not None:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.long)

        return item


class TransformerNetwork(nn.Module):
    """
    Neural Network architecture based on a pre-trained Transformer.
    Uses the [CLS] token representation for classification.
    """

    def __init__(self, model_name, num_classes=3):
        super(TransformerNetwork, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.config = self.encoder.config
        self.drop = nn.Dropout(0.1)
        self.fc = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token (index 0) from the last hidden state
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]
        output = self.fc(self.drop(cls_embedding))
        return output


class TransformerExpert:
    """
    The Neural Branch (Context Expert) of the ensemble.
    Wraps the training and inference logic for Transformer models.
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.device = Config.DEVICE
        self.best_model_state = None
        self.tokenizer = None

    def fit(self, train_texts, train_labels, val_texts, val_labels):
        """
        Trains the transformer model with early stopping.

        Args:
            train_texts: List/Array of training text.
            train_labels: List/Array of training labels (integers).
            val_texts: List/Array of validation text.
            val_labels: List/Array of validation labels (integers).
        """
        set_seed(Config.SEED)

        # Initialize Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Create Datasets
        train_dataset = TransformerDataset(
            texts=train_texts,
            targets=train_labels,
            tokenizer=self.tokenizer,
            max_len=Config.MAX_LEN,
        )
        val_dataset = TransformerDataset(
            texts=val_texts,
            targets=val_labels,
            tokenizer=self.tokenizer,
            max_len=Config.MAX_LEN,
        )

        # Create DataLoaders
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

        # Initialize Model
        model = TransformerNetwork(self.model_name, num_classes=3)
        model.to(self.device)

        # Optimization setup
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        num_train_steps = int(
            len(train_texts) / Config.TRAIN_BATCH_SIZE * Config.EPOCHS
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
            num_training_steps=num_train_steps,
        )

        criterion = nn.CrossEntropyLoss()

        # Training Loop
        best_loss = float("inf")
        patience_counter = 0

        print(f"Starting training for {self.model_name}...")

        for epoch in range(Config.EPOCHS):
            # --- Training ---
            model.train()
            train_losses = []

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = batch["target"].to(self.device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, targets)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

                optimizer.step()
                scheduler.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # --- Validation ---
            val_loss, val_preds = self._validate(model, val_loader, criterion)

            # Calculate metric using the utility function
            metric_score = calculate_log_loss(val_labels, val_preds)

            print(
                f"Epoch {epoch + 1}/{Config.EPOCHS} | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss (CE): {val_loss:.6f} | "
                f"Val Metric (LogLoss): {metric_score}"
            )

            # --- Early Stopping ---
            if metric_score < best_loss:
                best_loss = metric_score
                self.best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        # Load best model for future use
        if self.best_model_state is not None:
            print(f"Training finished. Best Metric: {best_loss}")
        else:
            # Fallback if training failed to improve (unlikely)
            self.best_model_state = model.state_dict()

        # Clean up to save memory
        del model
        del optimizer
        del scheduler
        torch.cuda.empty_cache()

    def _validate(self, model, dataloader, criterion):
        """Internal validation loop."""
        model.eval()
        val_losses = []
        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = batch["target"].to(self.device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, targets)
                val_losses.append(loss.item())

                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                all_preds.append(probs)

        avg_val_loss = np.mean(val_losses)
        all_preds = np.concatenate(all_preds, axis=0)

        return avg_val_loss, all_preds

    def predict_proba(self, texts):
        """
        Generates probability predictions for the given texts using the best trained model.

        Args:
            texts: List/Array of texts.

        Returns:
            np.ndarray: Predicted probabilities (n_samples, 3).
        """
        if self.best_model_state is None:
            raise ValueError("Model has not been trained yet.")

        # Initialize Tokenizer if not already done (e.g. if loading saved state directly)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Setup Data
        dataset = TransformerDataset(
            texts=texts, targets=None, tokenizer=self.tokenizer, max_len=Config.MAX_LEN
        )

        loader = DataLoader(
            dataset,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Load Model
        model = TransformerNetwork(self.model_name, num_classes=3)
        model.load_state_dict(self.best_model_state)
        model.to(self.device)
        model.eval()

        all_preds = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                all_preds.append(probs)

        # Clean up
        del model
        torch.cuda.empty_cache()

        return np.concatenate(all_preds, axis=0)
