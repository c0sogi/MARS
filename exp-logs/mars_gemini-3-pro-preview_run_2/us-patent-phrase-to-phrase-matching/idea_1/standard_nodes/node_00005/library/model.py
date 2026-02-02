import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from library.config import Config
from library.dataset import PhraseDataset
from library.utils import compute_pearson_correlation


class CrossEncoderRegressor:
    """
    Implements a Cross-Encoder for Semantic Textual Similarity.
    Fine-tunes a Transformer model to predict similarity scores directly.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.model_name = Config.MODEL_NAME

        # Initialize Tokenizer and Model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=1
        )
        self.model.to(self.device)

    def fit(self, train_df, val_df):
        """
        Fine-tunes the model on the training data.
        """
        # Create Datasets
        train_dataset = PhraseDataset(train_df, self.tokenizer)
        val_dataset = PhraseDataset(val_df, self.tokenizer)

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
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
            num_warmup_steps=int(total_steps * Config.WARMUP_RATIO),
            num_training_steps=total_steps,
        )

        loss_fn = nn.MSELoss()

        print(f"Starting training for {Config.EPOCHS} epochs...")

        for epoch in range(Config.EPOCHS):
            self.model.train()
            total_loss = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)

                loss = loss_fn(logits, labels)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # Validation Step
            val_corr = self.evaluate(val_loader, val_df["score"].values)
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Corr: {val_corr:.4f}"
            )

    def evaluate(self, dataloader, true_scores):
        self.model.eval()
        preds = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)
                preds.extend(logits.cpu().numpy())

        # Clip predictions
        preds = np.clip(preds, 0.0, 1.0)
        return compute_pearson_correlation(true_scores, preds)

    def predict(self, df):
        """
        Generates predictions for a dataframe.
        """
        dataset = PhraseDataset(df, self.tokenizer, is_test=True)
        dataloader = DataLoader(
            dataset,
            batch_size=Config.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        self.model.eval()
        preds = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)
                preds.extend(logits.cpu().numpy())

        return np.clip(preds, 0.0, 1.0)


def generate_submission(model, test_df):
    """
    Generates predictions for the test set and saves them to the submission file.
    """
    # Generate predictions
    preds = model.predict(test_df)

    # Create submission DataFrame
    submission = pd.DataFrame({"id": test_df["id"], "score": preds})

    # Save to CSV
    submission.to_csv(Config.SUBMISSION_FILE, index=False)
