import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)
import numpy as np
import os
import sys
from library.config import Config


class TokenClassifier(nn.Module):
    def __init__(self):
        super(TokenClassifier, self).__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(
            Config.MODEL_NAME, num_labels=Config.NUM_LABELS
        )

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return output


def compute_accuracy(logits, labels):
    """
    Computes accuracy ignoring -100 labels.
    """
    preds = torch.argmax(logits, dim=-1)
    mask = labels != -100
    correct = (preds == labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy.item(), mask.sum().item()


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        acc, n_tokens = compute_accuracy(logits, labels)
        total_correct += acc * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, avg_acc


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            acc, n_tokens = compute_accuracy(logits, labels)
            total_correct += acc * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, avg_acc


def train_model(train_dataset, val_dataset):
    """
    Main training loop with Early Stopping.
    """
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

    device = Config.DEVICE
    model = TokenClassifier()
    model.to(device)

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # Early Stopping variables
    best_val_loss = float("inf")
    patience = 2
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print(f"Train Loss: {train_loss}")
        print(f"Train Acc: {train_acc}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Acc: {val_acc}")

        # Checkpoint and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print("Saved best model.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model for return
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    return model


def predict_labels(model, test_dataset):
    """
    Runs inference on the test dataset and returns a flat list of predicted labels
    corresponding to the original tokens.
    """
    device = Config.DEVICE
    model.to(device)
    model.eval()

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    all_predictions = []

    # We need to reconstruct the predictions per sentence to flatten them correctly later
    # The dataset is grouped by sentence.

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_ids_batch = batch["word_ids"]  # CPU tensor

            outputs = model(input_ids, attention_mask)
            logits = outputs.logits  # (Batch, Seq, NumLabels)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            # Decode sub-word predictions to word-level predictions
            batch_size = input_ids.shape[0]

            for i in range(batch_size):
                word_ids = word_ids_batch[i].numpy()
                pred_ids = preds[i]

                sentence_preds = []
                seen_words = set()

                # Iterate through the sequence
                for idx, word_id in enumerate(word_ids):
                    if word_id == -1:
                        continue

                    # We take the prediction of the first sub-token of a word
                    if word_id not in seen_words:
                        label_id = pred_ids[idx]
                        label_str = Config.ID2LABEL[label_id]
                        sentence_preds.append(label_str)
                        seen_words.add(word_id)

                # Append the list of labels for this sentence
                all_predictions.append(sentence_preds)

    return all_predictions
