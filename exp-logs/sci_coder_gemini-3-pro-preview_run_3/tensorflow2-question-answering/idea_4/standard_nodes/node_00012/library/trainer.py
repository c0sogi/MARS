import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time

from library.config import Config
from library.models import SiameseRanker, SeparableConvReader
from library.dataset import (
    NQRankerDataset,
    ranker_collate_fn,
    NQReaderDataset,
    reader_collate_fn,
)


class RankerTrainer:
    """
    Trainer for the SiameseRanker model using Margin Ranking Loss.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.DEVICE)

        # Model
        self.model = SiameseRanker(config).to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Loss Function: Margin Ranking Loss
        # Loss(x1, x2, y) = max(0, -y * (x1 - x2) + margin)
        # We want x1 (pos) > x2 (neg), so y=1.
        self.criterion = nn.MarginRankingLoss(margin=1.0)

    def train(self, load_cached_data=True):
        print(f"Initializing Ranker Training on {self.device}...")

        # Data Loaders
        train_dataset = NQRankerDataset(
            self.config, split="train", load_cached_data=load_cached_data
        )
        val_dataset = NQRankerDataset(
            self.config, split="val", load_cached_data=load_cached_data
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            collate_fn=ranker_collate_fn,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            collate_fn=ranker_collate_fn,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.EPOCHS):
            start_time = time.time()
            self.model.train()
            total_train_loss = 0.0

            for batch_idx, (q_batch, pos_batch, neg_batch) in enumerate(train_loader):
                q_batch = q_batch.to(self.device)
                pos_batch = pos_batch.to(self.device)
                neg_batch = neg_batch.to(self.device)

                # neg_batch shape: (B, Num_Neg, L)
                batch_size, num_neg, seq_len = neg_batch.shape

                self.optimizer.zero_grad()

                # 1. Score Positive Pairs
                pos_scores = self.model(q_batch, pos_batch)  # (B,)

                # 2. Score Negative Pairs
                # We need to compute score for each negative against the question.
                # Expand Q to match flattened negatives
                q_expanded = q_batch.repeat_interleave(num_neg, dim=0)  # (B*Num_Neg, L)
                neg_flat = neg_batch.view(-1, seq_len)  # (B*Num_Neg, L)

                neg_scores_flat = self.model(q_expanded, neg_flat)  # (B*Num_Neg,)

                # 3. Compute Loss
                # Expand pos_scores to match neg_scores_flat
                pos_scores_expanded = pos_scores.repeat_interleave(num_neg, dim=0)

                # Targets are all 1 because we want pos > neg
                targets = torch.ones_like(pos_scores_expanded).to(self.device)

                loss = self.criterion(pos_scores_expanded, neg_scores_flat, targets)

                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation
            val_loss, val_acc = self.validate(val_loader)

            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

            print(f"Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s")
            print(f"\tTrain Loss: {avg_train_loss:.6f}")
            print(f"\t Val. Loss: {val_loss:.6f} | Val. Acc: {val_acc:.6f}%")

            # Early Stopping & Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.config.RANKER_MODEL_PATH)
                print(
                    f"\tValidation loss decreased. Saving model to {self.config.RANKER_MODEL_PATH}"
                )
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"\tEarlyStopping counter: {patience_counter} out of {self.config.PATIENCE}"
                )
                if patience_counter >= self.config.PATIENCE:
                    print("Early stopping triggered.")
                    break

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        correct_rankings = 0
        total_comparisons = 0

        with torch.no_grad():
            for q_batch, pos_batch, neg_batch in dataloader:
                q_batch = q_batch.to(self.device)
                pos_batch = pos_batch.to(self.device)
                neg_batch = neg_batch.to(self.device)

                batch_size, num_neg, seq_len = neg_batch.shape

                pos_scores = self.model(q_batch, pos_batch)

                q_expanded = q_batch.repeat_interleave(num_neg, dim=0)
                neg_flat = neg_batch.view(-1, seq_len)
                neg_scores_flat = self.model(q_expanded, neg_flat)

                pos_scores_expanded = pos_scores.repeat_interleave(num_neg, dim=0)
                targets = torch.ones_like(pos_scores_expanded).to(self.device)

                loss = self.criterion(pos_scores_expanded, neg_scores_flat, targets)
                total_loss += loss.item()

                # Calculate Accuracy: How often is pos_score > neg_score?
                correct_rankings += (pos_scores_expanded > neg_scores_flat).sum().item()
                total_comparisons += pos_scores_expanded.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = (
            (correct_rankings / total_comparisons) * 100.0
            if total_comparisons > 0
            else 0.0
        )

        return avg_loss, accuracy


class ReaderTrainer:
    """
    Trainer for the SeparableConvReader model using Cross Entropy Loss.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.DEVICE)

        # Model
        self.model = SeparableConvReader(config).to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Loss Function
        self.criterion = nn.CrossEntropyLoss()

    def train(self, load_cached_data=True):
        print(f"Initializing Reader Training on {self.device}...")

        # Data Loaders
        train_dataset = NQReaderDataset(
            self.config, split="train", load_cached_data=load_cached_data
        )
        val_dataset = NQReaderDataset(
            self.config, split="val", load_cached_data=load_cached_data
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            collate_fn=reader_collate_fn,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            collate_fn=reader_collate_fn,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.EPOCHS):
            start_time = time.time()
            self.model.train()
            total_train_loss = 0.0

            for batch_idx, (input_ids, start_targets, end_targets) in enumerate(
                train_loader
            ):
                input_ids = input_ids.to(self.device)
                start_targets = start_targets.to(self.device)
                end_targets = end_targets.to(self.device)

                self.optimizer.zero_grad()

                # Forward
                start_logits, end_logits = self.model(input_ids)

                # Loss
                loss_start = self.criterion(start_logits, start_targets)
                loss_end = self.criterion(end_logits, end_targets)
                loss = (loss_start + loss_end) / 2.0

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation
            val_loss, val_em = self.validate(val_loader)

            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

            print(f"Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s")
            print(f"\tTrain Loss: {avg_train_loss:.6f}")
            print(f"\t Val. Loss: {val_loss:.6f} | Val. EM: {val_em:.6f}%")

            # Early Stopping & Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.config.READER_MODEL_PATH)
                print(
                    f"\tValidation loss decreased. Saving model to {self.config.READER_MODEL_PATH}"
                )
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"\tEarlyStopping counter: {patience_counter} out of {self.config.PATIENCE}"
                )
                if patience_counter >= self.config.PATIENCE:
                    print("Early stopping triggered.")
                    break

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        exact_matches = 0
        total_samples = 0

        with torch.no_grad():
            for input_ids, start_targets, end_targets in dataloader:
                input_ids = input_ids.to(self.device)
                start_targets = start_targets.to(self.device)
                end_targets = end_targets.to(self.device)

                start_logits, end_logits = self.model(input_ids)

                loss_start = self.criterion(start_logits, start_targets)
                loss_end = self.criterion(end_logits, end_targets)
                loss = (loss_start + loss_end) / 2.0
                total_loss += loss.item()

                # Predictions
                start_preds = torch.argmax(start_logits, dim=1)
                end_preds = torch.argmax(end_logits, dim=1)

                # Exact Match Calculation
                match = (start_preds == start_targets) & (end_preds == end_targets)
                exact_matches += match.sum().item()
                total_samples += input_ids.size(0)

        avg_loss = total_loss / len(dataloader)
        em_score = (exact_matches / total_samples) * 100.0 if total_samples > 0 else 0.0

        return avg_loss, em_score
