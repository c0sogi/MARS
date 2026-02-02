import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random

from library.utils import ensure_dir, load_embeddings, CACHE_DIR, get_dataset_partitions
from library.models import CompareAggregateRanker, DilatedConvReader
from library.data_loader import (
    get_tokenizer,
    process_ranker_data,
    process_reader_data,
    NQRankerDataset,
    NQReaderDataset,
    ranker_collate_fn,
    reader_collate_fn,
)

# Set seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class ModelTrainer:
    def __init__(self, cache_dir=CACHE_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        ensure_dir(self.cache_dir)

        # Loss functions
        self.ranker_criterion = nn.BCEWithLogitsLoss()
        self.reader_criterion = nn.CrossEntropyLoss()

    def save_model(self, model, filename):
        path = os.path.join(self.cache_dir, filename)
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")

    def train_ranker(
        self, model, train_loader, val_loader, epochs=5, lr=1e-3, patience=2
    ):
        print(f"\nStarting Ranker Training on {self.device}...")
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Training Phase
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (q_ids, p_ids, labels) in enumerate(train_loader):
                q_ids, p_ids, labels = (
                    q_ids.to(self.device),
                    p_ids.to(self.device),
                    labels.to(self.device),
                )

                optimizer.zero_grad()
                logits = model(q_ids, p_ids)
                loss = self.ranker_criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # Metrics
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            avg_train_loss = train_loss / len(train_loader)
            train_acc = correct / total

            # Validation Phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for q_ids, p_ids, labels in val_loader:
                    q_ids, p_ids, labels = (
                        q_ids.to(self.device),
                        p_ids.to(self.device),
                        labels.to(self.device),
                    )
                    logits = model(q_ids, p_ids)
                    loss = self.ranker_criterion(logits, labels)
                    val_loss += loss.item()

                    preds = (torch.sigmoid(logits) > 0.5).float()
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {avg_train_loss} | Train Acc: {train_acc} | "
                f"Val Loss: {avg_val_loss} | Val Acc: {val_acc}"
            )

            # Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model(model, "ranker_best.pth")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered for Ranker.")
                    break

        return model

    def train_reader(
        self, model, train_loader, val_loader, epochs=5, lr=1e-3, patience=2
    ):
        print(f"\nStarting Reader Training on {self.device}...")
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Training Phase
            model.train()
            train_loss = 0.0

            for batch_idx, (input_ids, start_targets, end_targets) in enumerate(
                train_loader
            ):
                input_ids = input_ids.to(self.device)
                start_targets = start_targets.to(self.device)
                end_targets = end_targets.to(self.device)

                optimizer.zero_grad()
                start_logits, end_logits = model(input_ids)

                loss_start = self.reader_criterion(start_logits, start_targets)
                loss_end = self.reader_criterion(end_logits, end_targets)
                loss = loss_start + loss_end

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation Phase
            model.eval()
            val_loss = 0.0
            exact_match = 0
            total = 0

            with torch.no_grad():
                for input_ids, start_targets, end_targets in val_loader:
                    input_ids = input_ids.to(self.device)
                    start_targets = start_targets.to(self.device)
                    end_targets = end_targets.to(self.device)

                    start_logits, end_logits = model(input_ids)

                    loss_start = self.reader_criterion(start_logits, start_targets)
                    loss_end = self.reader_criterion(end_logits, end_targets)
                    val_loss += (loss_start + loss_end).item()

                    # Calculate Exact Match
                    pred_start = torch.argmax(start_logits, dim=1)
                    pred_end = torch.argmax(end_logits, dim=1)

                    match = (pred_start == start_targets) & (pred_end == end_targets)
                    exact_match += match.sum().item()
                    total += input_ids.size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_em = exact_match / total

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {avg_train_loss} | "
                f"Val Loss: {avg_val_loss} | Val Exact Match: {val_em}"
            )

            # Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model(model, "reader_best.pth")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered for Reader.")
                    break

        return model


def run_training_pipeline(
    sample_size=None,
    ranker_epochs=5,
    reader_epochs=5,
    batch_size=32,
    load_cached_data=True,
):
    """
    Orchestrates the data loading and training process for both models.
    """
    trainer = ModelTrainer()

    # 1. Load Metadata
    print("Loading metadata...")
    train_meta, val_meta, _ = get_dataset_partitions()
    if train_meta.empty or val_meta.empty:
        print("Metadata not found. Please ensure metadata generation script has run.")
        return

    # 2. Tokenizer
    tokenizer = get_tokenizer(
        train_meta, load_cached_data=load_cached_data, sample_size=sample_size
    )
    vocab_size = tokenizer.vocab_count
    print(f"Vocabulary size: {vocab_size}")

    # 3. Embeddings
    embedding_matrix = load_embeddings(
        tokenizer.word_index, load_cached_data=load_cached_data
    )

    # ---------------------------------------------------------
    # Train Ranker
    # ---------------------------------------------------------
    print("\nPreparing Ranker Data...")
    ranker_train_df = process_ranker_data(
        train_meta,
        tokenizer,
        load_cached_data=load_cached_data,
        split="train",
        sample_size=sample_size,
    )
    ranker_val_df = process_ranker_data(
        val_meta,
        tokenizer,
        load_cached_data=load_cached_data,
        split="val",
        sample_size=sample_size,
    )

    ranker_train_loader = DataLoader(
        NQRankerDataset(ranker_train_df),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ranker_collate_fn,
    )
    ranker_val_loader = DataLoader(
        NQRankerDataset(ranker_val_df),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ranker_collate_fn,
    )

    ranker_model = CompareAggregateRanker(embedding_matrix)
    trainer.train_ranker(
        ranker_model, ranker_train_loader, ranker_val_loader, epochs=ranker_epochs
    )

    # Free up memory
    del (
        ranker_model,
        ranker_train_loader,
        ranker_val_loader,
        ranker_train_df,
        ranker_val_df,
    )
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Train Reader
    # ---------------------------------------------------------
    print("\nPreparing Reader Data...")
    reader_train_df = process_reader_data(
        train_meta,
        tokenizer,
        load_cached_data=load_cached_data,
        split="train",
        sample_size=sample_size,
    )
    reader_val_df = process_reader_data(
        val_meta,
        tokenizer,
        load_cached_data=load_cached_data,
        split="val",
        sample_size=sample_size,
    )

    reader_train_loader = DataLoader(
        NQReaderDataset(reader_train_df),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=reader_collate_fn,
    )
    reader_val_loader = DataLoader(
        NQReaderDataset(reader_val_df),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=reader_collate_fn,
    )

    reader_model = DilatedConvReader(embedding_matrix)
    trainer.train_reader(
        reader_model, reader_train_loader, reader_val_loader, epochs=reader_epochs
    )

    print("\nTraining pipeline completed.")
