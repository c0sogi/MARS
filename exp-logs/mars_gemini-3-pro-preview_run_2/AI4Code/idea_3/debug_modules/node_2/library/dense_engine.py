import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.data_factory import MarkdownRankDataset
from library.utils import seed_everything


class TransformerRanker(nn.Module):
    """
    Neural Network module combining a Transformer backbone with a linear regression head.
    """

    def __init__(self, model_name):
        super(TransformerRanker, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.config = self.backbone.config

        # Linear head to project pooled embedding to a scalar rank
        self.regressor = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # Forward pass through backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Use the representation of the [CLS] token (index 0)
        # Some models return 'pooler_output', others 'last_hidden_state'
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # Fallback for models without pooler (e.g. DistilBERT)
            pooled_output = outputs.last_hidden_state[:, 0, :]

        # Project to scalar
        logits = self.regressor(pooled_output)
        return logits


class DenseEngine:
    """
    Manages the training and inference of the Dense Semantic Stream (Transformer).
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(Config.TRANSFORMER_MODEL_NAME)
        self.model = TransformerRanker(Config.TRANSFORMER_MODEL_NAME)
        self.model.to(self.device)

    def fit(self, df_train, df_val=None, patience=1):
        """
        Trains the TransformerRanker model.

        Args:
            df_train (pd.DataFrame): Training data.
            df_val (pd.DataFrame, optional): Validation data.
            patience (int): Early stopping patience.
        """
        print(f"Fitting DenseEngine on device: {self.device}")
        seed_everything(Config.SEED)

        # Prepare Datasets and Loaders
        train_dataset = MarkdownRankDataset(df_train, self.tokenizer, Config.MAX_LEN)
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        val_loader = None
        if df_val is not None:
            val_dataset = MarkdownRankDataset(df_val, self.tokenizer, Config.MAX_LEN)
            val_loader = DataLoader(
                val_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

        # Optimizer and Scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        total_steps = len(train_loader) * Config.EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        loss_fn = nn.MSELoss()
        scaler = GradScaler(enabled=Config.FP16)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            self.model.train()
            train_loss_accum = 0.0

            # Training Loop
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = batch["label"].to(self.device).unsqueeze(1)  # [Batch, 1]

                optimizer.zero_grad()

                with autocast(enabled=Config.FP16):
                    outputs = self.model(input_ids, attention_mask)
                    loss = loss_fn(outputs, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                train_loss_accum += loss.item()

            avg_train_loss = train_loss_accum / len(train_loader)
            print(f"Epoch {epoch + 1}/{Config.EPOCHS} - Train MSE: {avg_train_loss}")

            # Validation Loop
            if val_loader:
                avg_val_loss = self.evaluate(val_loader, loss_fn)
                print(f"Epoch {epoch + 1}/{Config.EPOCHS} - Val MSE: {avg_val_loss}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    self.save()  # Save best model
                else:
                    patience_counter += 1
                    print(
                        f"Validation loss did not improve. Patience: {patience_counter}/{patience}"
                    )
                    if patience_counter >= patience:
                        print("Early stopping triggered.")
                        break
            else:
                # If no validation set, save at end of every epoch
                self.save()

        # Load best model for future use if validation was used
        if val_loader:
            self.load()

    def evaluate(self, dataloader, loss_fn):
        self.model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = batch["label"].to(self.device).unsqueeze(1)

                with autocast(enabled=Config.FP16):
                    outputs = self.model(input_ids, attention_mask)
                    loss = loss_fn(outputs, targets)

                val_loss_accum += loss.item()

        return val_loss_accum / len(dataloader)

    def predict(self, df):
        """
        Generates rank predictions for the provided data.

        Args:
            df (pd.DataFrame): Data containing 'source' column.

        Returns:
            np.ndarray: Predicted ranks.
        """
        dataset = MarkdownRankDataset(df, self.tokenizer, Config.MAX_LEN)
        dataloader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE * 2,  # Larger batch size for inference
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                with autocast(enabled=Config.FP16):
                    outputs = self.model(input_ids, attention_mask)

                # Move to CPU and flatten
                preds = outputs.cpu().numpy().flatten()
                predictions.append(preds)

        return np.concatenate(predictions)

    def save(self, output_dir=None):
        """
        Saves the model weights and tokenizer.
        """
        if output_dir is None:
            output_dir = Config.WORKING_DIR
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving DenseEngine artifacts to {output_dir}...")
        torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
        self.tokenizer.save_pretrained(output_dir)

    def load(self, input_dir=None):
        """
        Loads the model weights and tokenizer.
        """
        if input_dir is None:
            input_dir = Config.WORKING_DIR
        print(f"Loading DenseEngine artifacts from {input_dir}...")
        if os.path.exists(Config.BEST_MODEL_PATH):
            self.model.load_state_dict(
                torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
            )
        else:
            print(
                f"Warning: Model file not found at {Config.BEST_MODEL_PATH}. Using initialized weights."
            )

        # Tokenizer is loaded in __init__ from pretrained, but if we saved custom vocab we could reload it
        # For this task, we assume standard pretrained tokenizer is sufficient or saved in dir
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(input_dir)
        except:
            pass  # Fallback to the one loaded in __init__
