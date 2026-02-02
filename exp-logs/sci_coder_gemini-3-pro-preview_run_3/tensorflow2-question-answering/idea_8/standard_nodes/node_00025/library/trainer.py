import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os

from library.config import Config
from library.utils import load_glove_embeddings
from library.data import (
    get_vocabulary,
    get_ranker_data,
    get_reader_data,
    NQRankerDataset,
    NQReaderDataset,
)
from library.models import ANBoWRanker, ConvBiDAFReader


class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Trainer initialized on device: {self.device}")

        # Load Vocabulary
        self.vocab = get_vocabulary(load_cached_data=True)

        # Load Embeddings
        # Note: In a real scenario, one might point to a glove file, but here we rely on the util's random init or cache.
        self.embedding_matrix = load_glove_embeddings(
            self.vocab.token_to_idx, Config.EMBEDDING_DIM, load_cached_data=True
        )

    def _get_dataloader(self, dataset_cls, data_df, shuffle=True):
        dataset = dataset_cls(data_df, self.vocab)
        return DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if self.device.type == "cuda" else False,
        )

    def train_ranker(self):
        print("\n--- Starting Ranker Training ---")

        # Data Loading
        train_df = get_ranker_data(split="train", load_cached_data=True)
        val_df = get_ranker_data(split="val", load_cached_data=True)

        train_loader = self._get_dataloader(NQRankerDataset, train_df, shuffle=True)
        val_loader = self._get_dataloader(NQRankerDataset, val_df, shuffle=False)

        # Model Initialization
        model = ANBoWRanker(embedding_matrix=self.embedding_matrix).to(self.device)

        # Optimization
        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            # Training Phase
            model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                q_ids = batch["q_ids"].to(self.device)
                c_ids = batch["c_ids"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                logits = model(q_ids, c_ids)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * labels.size(0)

                # Accuracy
                preds = (torch.sigmoid(logits) > 0.5).float()
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            avg_train_loss = train_loss_sum / train_total
            train_acc = train_correct / train_total

            # Validation Phase
            model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    q_ids = batch["q_ids"].to(self.device)
                    c_ids = batch["c_ids"].to(self.device)
                    labels = batch["label"].to(self.device)

                    logits = model(q_ids, c_ids)
                    loss = criterion(logits, labels)

                    val_loss_sum += loss.item() * labels.size(0)
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            avg_val_loss = val_loss_sum / val_total
            val_acc = val_correct / val_total

            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
                f"Train Loss: {avg_train_loss} | Train Acc: {train_acc} | "
                f"Val Loss: {avg_val_loss} | Val Acc: {val_acc}"
            )

            # Early Stopping and Checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), Config.RANKER_MODEL_PATH)
                print(f"New best ranker model saved to {Config.RANKER_MODEL_PATH}")
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

    def train_reader(self):
        print("\n--- Starting Reader Training ---")

        # Data Loading
        train_df = get_reader_data(split="train", load_cached_data=True)
        val_df = get_reader_data(split="val", load_cached_data=True)

        train_loader = self._get_dataloader(NQReaderDataset, train_df, shuffle=True)
        val_loader = self._get_dataloader(NQReaderDataset, val_df, shuffle=False)

        # Model Initialization
        model = ConvBiDAFReader(embedding_matrix=self.embedding_matrix).to(self.device)

        # Optimization
        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            # Training Phase
            model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                q_ids = batch["q_ids"].to(self.device)
                c_ids = batch["c_ids"].to(self.device)
                start_targets = batch["start_idx"].to(self.device)
                end_targets = batch["end_idx"].to(self.device)

                optimizer.zero_grad()
                start_logits, end_logits = model(q_ids, c_ids)

                loss_start = criterion(start_logits, start_targets)
                loss_end = criterion(end_logits, end_targets)
                loss = loss_start + loss_end

                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * q_ids.size(0)

                # Exact Match Accuracy
                pred_start = torch.argmax(start_logits, dim=1)
                pred_end = torch.argmax(end_logits, dim=1)
                correct = (
                    ((pred_start == start_targets) & (pred_end == end_targets))
                    .sum()
                    .item()
                )
                train_correct += correct
                train_total += q_ids.size(0)

            avg_train_loss = train_loss_sum / train_total
            train_acc = train_correct / train_total

            # Validation Phase
            model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    q_ids = batch["q_ids"].to(self.device)
                    c_ids = batch["c_ids"].to(self.device)
                    start_targets = batch["start_idx"].to(self.device)
                    end_targets = batch["end_idx"].to(self.device)

                    start_logits, end_logits = model(q_ids, c_ids)

                    loss_start = criterion(start_logits, start_targets)
                    loss_end = criterion(end_logits, end_targets)
                    loss = loss_start + loss_end

                    val_loss_sum += loss.item() * q_ids.size(0)

                    pred_start = torch.argmax(start_logits, dim=1)
                    pred_end = torch.argmax(end_logits, dim=1)
                    correct = (
                        ((pred_start == start_targets) & (pred_end == end_targets))
                        .sum()
                        .item()
                    )
                    val_correct += correct
                    val_total += q_ids.size(0)

            avg_val_loss = val_loss_sum / val_total
            val_acc = val_correct / val_total

            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
                f"Train Loss: {avg_train_loss} | Train EM Acc: {train_acc} | "
                f"Val Loss: {avg_val_loss} | Val EM Acc: {val_acc}"
            )

            # Early Stopping and Checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), Config.READER_MODEL_PATH)
                print(f"New best reader model saved to {Config.READER_MODEL_PATH}")
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break
