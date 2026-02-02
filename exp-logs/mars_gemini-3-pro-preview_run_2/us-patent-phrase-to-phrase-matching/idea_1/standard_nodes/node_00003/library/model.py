import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from library.config import Config
from library.data_manager import PatentDataset
from library.utils import compute_pearson_correlation


class CrossEncoderModel:
    """
    Implements a Cross-Encoder model using a pre-trained Transformer.
    The model takes pairs of (anchor, target) and predicts a similarity score.
    Cite solution_lesson_node_00002: Moving from Bi-Encoder to Cross-Encoder to capture deeper signal.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            Config.MODEL_NAME, num_labels=1
        )
        self.model.to(self.device)

    def fit(self, train_df, val_df):
        """
        Fine-tunes the transformer model.
        """
        # Prepare DataLoaders
        train_dataset = PatentDataset(train_df, self.tokenizer, Config.MAX_LENGTH)
        val_dataset = PatentDataset(val_df, self.tokenizer, Config.MAX_LENGTH)

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Optimization
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        num_training_steps = len(train_loader) * Config.EPOCHS
        num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        loss_fn = nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler()

        print(f"Starting training for {Config.EPOCHS} epochs...")

        best_val_corr = -1.0

        for epoch in range(Config.EPOCHS):
            self.model.train()
            train_loss = 0

            for batch in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}", leave=False
            ):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                    logits = outputs.logits.squeeze(-1)
                    loss = loss_fn(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            val_preds = self._predict_loader(val_loader)
            val_corr = compute_pearson_correlation(val_df["score"].values, val_preds)

            print(
                f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} - Val Corr: {val_corr:.4f}"
            )

            if val_corr > best_val_corr:
                best_val_corr = val_corr
                # In a real scenario, we would save the model checkpoint here
                # torch.save(self.model.state_dict(), "best_model.pth")

    def predict(self, df):
        """
        Generates predictions for a dataframe.
        """
        dataset = PatentDataset(df, self.tokenizer, Config.MAX_LENGTH, is_test=True)
        loader = DataLoader(
            dataset,
            batch_size=Config.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        return self._predict_loader(loader)

    def _predict_loader(self, loader):
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                    logits = outputs.logits.squeeze(-1)

                all_preds.append(logits.cpu().numpy())

        preds = np.concatenate(all_preds)
        # Clip predictions to valid range [0, 1]
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
