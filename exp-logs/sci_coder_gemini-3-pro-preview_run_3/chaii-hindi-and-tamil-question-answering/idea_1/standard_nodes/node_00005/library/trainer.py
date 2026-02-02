import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer

from library.config import Config
from library.utils import jaccard
from library.model import QATokenClassifier
from library.data_loader import prepare_data


class Trainer:
    """
    Trainer class for the Question Answering model.
    """

    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def train_epoch(self, train_loader, optimizer, scheduler, epoch):
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def validate(self, val_loader, val_df):
        self.model.eval()

        # Store all predictions: {sample_idx: (score, text)}
        all_preds = {}

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                sample_idxs = batch["sample_idx"].cpu().numpy()

                outputs = self.model(input_ids, attention_mask)
                logits = outputs.logits

                decoded_results = self._decode_batch(logits, input_ids)

                for k, (text, score) in enumerate(decoded_results):
                    s_idx = sample_idxs[k]

                    # Keep the prediction with the highest score for this sample
                    if s_idx not in all_preds or score > all_preds[s_idx][0]:
                        all_preds[s_idx] = (score, text)

        # Calculate Jaccard
        total_jaccard = 0.0
        count = 0

        # Iterate over the original dataframe to ensure we cover all examples
        for idx, row in val_df.iterrows():
            gt = row["answer_text"]
            if idx in all_preds:
                pred = all_preds[idx][1]
            else:
                pred = ""

            total_jaccard += jaccard(gt, pred)
            count += 1

        return total_jaccard / count if count > 0 else 0.0

    def predict(self, test_loader, test_df):
        self.model.eval()

        # Store all predictions: {sample_idx: (score, text)}
        all_preds = {}

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                sample_idxs = batch["sample_idx"].cpu().numpy()

                outputs = self.model(input_ids, attention_mask)
                logits = outputs.logits

                decoded_results = self._decode_batch(logits, input_ids)

                for k, (text, score) in enumerate(decoded_results):
                    s_idx = sample_idxs[k]
                    if s_idx not in all_preds or score > all_preds[s_idx][0]:
                        all_preds[s_idx] = (score, text)

        # Construct final lists aligned with test_df
        ids = []
        predictions = []

        for idx, row in test_df.iterrows():
            ids.append(row["id"])
            if idx in all_preds:
                predictions.append(all_preds[idx][1])
            else:
                predictions.append("")

        return ids, predictions

    def _decode_batch(self, logits, input_ids):
        # Calculate probabilities
        probs = torch.softmax(logits, dim=2).cpu().numpy()
        preds = torch.argmax(logits, dim=2).cpu().numpy()
        input_ids_np = input_ids.cpu().numpy()

        results = []  # List of (text, score)

        B_ANS = Config.LABELS_TO_IDS["B-ANS"]
        I_ANS = Config.LABELS_TO_IDS["I-ANS"]

        for k in range(len(preds)):
            token_ids = input_ids_np[k]
            tags = preds[k]
            token_probs = probs[k]

            candidates = []
            i = 0
            while i < len(tags):
                if tags[i] == B_ANS:
                    start = i
                    # Extend span while tags are I-ANS
                    j = i + 1
                    while j < len(tags) and tags[j] == I_ANS:
                        j += 1
                    end = j - 1

                    # Decode text
                    span = token_ids[start : end + 1]
                    pred_str = self.tokenizer.decode(
                        span, skip_special_tokens=True
                    ).strip()

                    # Score: Mean probability of the tokens in the span
                    span_probs = [
                        token_probs[x, tags[x]] for x in range(start, end + 1)
                    ]
                    score = np.mean(span_probs)

                    candidates.append((pred_str, score))

                    # Move index to end of this span
                    i = j
                else:
                    i += 1

            # Select the candidate with the highest score
            if candidates:
                # Sort by score descending
                best_candidate = sorted(candidates, key=lambda x: x[1], reverse=True)[0]
                results.append(best_candidate)
            else:
                results.append(("", 0.0))

        return results

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model.load(path, self.device)


def run_training():
    Config.setup()

    # Load Data
    train_dataset, val_dataset, test_dataset = prepare_data(load_cached_data=True)

    # Load DataFrames for alignment
    df_val = pd.read_csv(Config.VAL_DATA_PATH)
    df_test = pd.read_csv(Config.TEST_DATA_PATH)

    if Config.DEBUG:
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Setup Model
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = QATokenClassifier(Config.MODEL_NAME)

    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_training_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    trainer = Trainer(model, tokenizer, Config.DEVICE)

    print(f"Starting training on device: {Config.DEVICE}")

    best_val_score = -1.0
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        train_loss = trainer.train_epoch(train_loader, optimizer, scheduler, epoch)
        val_score = trainer.validate(val_loader, df_val)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val Jaccard: {val_score}"
        )

        if val_score > best_val_score:
            best_val_score = val_score
            trainer.save_model(Config.MODEL_SAVE_PATH)
            patience_counter = 0
            print(f"New best model saved with Jaccard: {best_val_score}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    # Prediction
    print("Loading best model for prediction...")
    trainer.load_model(Config.MODEL_SAVE_PATH)

    ids, predictions = trainer.predict(test_loader, df_test)

    submission_df = pd.DataFrame({"id": ids, "PredictionString": predictions})

    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
