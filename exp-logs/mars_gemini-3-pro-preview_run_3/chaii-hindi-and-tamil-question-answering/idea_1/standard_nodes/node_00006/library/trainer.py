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

        # Dictionary to aggregate predictions: {sample_idx: [(pred_str, score), ...]}
        sample_preds = {}

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                sample_indices = batch["sample_idx"].tolist()

                outputs = self.model(input_ids, attention_mask)
                logits = outputs.logits

                # Decode with scores
                pred_results = self._decode_batch(logits, input_ids)

                for idx, (pred, score) in zip(sample_indices, pred_results):
                    if idx not in sample_preds:
                        sample_preds[idx] = []
                    sample_preds[idx].append((pred, score))

        # Compute Jaccard
        total_jaccard = 0.0
        count = 0

        # Iterate over the original dataframe
        for idx, row in val_df.iterrows():
            gt = row["answer_text"]
            candidates = sample_preds.get(idx, [])

            # Aggregation Strategy: Pick prediction with highest confidence score
            # Cite solution_lesson_node_00004: Use model confidence scores
            best_pred = ""
            if candidates:
                # Sort by score descending
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_pred = candidates[0][0]

            score = jaccard(gt, best_pred)
            total_jaccard += score
            count += 1

        return total_jaccard / count if count > 0 else 0.0

    def predict(self, test_loader, test_df):
        self.model.eval()
        sample_preds = {}

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                sample_indices = batch["sample_idx"].tolist()

                outputs = self.model(input_ids, attention_mask)
                logits = outputs.logits

                pred_results = self._decode_batch(logits, input_ids)

                for idx, (pred, score) in zip(sample_indices, pred_results):
                    if idx not in sample_preds:
                        sample_preds[idx] = []
                    sample_preds[idx].append((pred, score))

        # Aggregate and align with test_df
        ids = []
        predictions = []

        for idx, row in test_df.iterrows():
            ids.append(row["id"])
            candidates = sample_preds.get(idx, [])

            best_pred = ""
            if candidates:
                # Sort by score descending
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_pred = candidates[0][0]

            predictions.append(best_pred)

        return ids, predictions

    def _decode_batch(self, logits, input_ids):
        # Calculate probabilities
        probs = torch.softmax(logits, dim=2)
        confidences, preds = torch.max(probs, dim=2)

        preds = preds.cpu().numpy()
        confidences = confidences.cpu().numpy()
        input_ids_np = input_ids.cpu().numpy()

        decoded_results = []

        B_ANS = Config.LABELS_TO_IDS["B-ANS"]
        I_ANS = Config.LABELS_TO_IDS["I-ANS"]

        for k in range(len(preds)):
            token_ids = input_ids_np[k]
            tags = preds[k]
            confs = confidences[k]

            start = -1
            end = -1

            # Find first B-ANS
            for idx, tag in enumerate(tags):
                if tag == B_ANS:
                    start = idx
                    break

            if start != -1:
                end = start
                for idx in range(start + 1, len(tags)):
                    if tags[idx] == I_ANS:
                        end = idx
                    else:
                        break

                span = token_ids[start : end + 1]
                pred_str = self.tokenizer.decode(span, skip_special_tokens=True).strip()

                # Calculate score: mean confidence of the span tokens
                span_confs = confs[start : end + 1]
                score = np.mean(span_confs)
            else:
                pred_str = ""
                score = 0.0

            decoded_results.append((pred_str, score))

        return decoded_results

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
