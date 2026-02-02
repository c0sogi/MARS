import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config


class RankerTrainer:
    """
    Trainer for the Histogram-Based Matching Ranker.
    Optimizes using Pairwise Hinge Loss (MarginRankingLoss).
    """

    def __init__(self, model, device=Config.DEVICE):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MarginRankingLoss(margin=Config.RANKER_MARGIN)
        self.best_val_acc = -1.0

    def train_epoch(self, dataloader, optimizer):
        self.model.train()
        total_loss = 0.0
        correct_pairs = 0
        total_pairs = 0

        for batch in dataloader:
            # Unpack batch (collated by ranker_collate_fn)
            q_ids, pos_ids, neg_ids = [t.to(self.device) for t in batch]

            optimizer.zero_grad()

            # Forward pass
            pos_scores = self.model(q_ids, pos_ids)
            neg_scores = self.model(q_ids, neg_ids)

            # Target for MarginRankingLoss: 1 means pos_scores should be > neg_scores
            targets = torch.ones(pos_scores.size(), device=self.device)

            loss = self.criterion(pos_scores, neg_scores, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * q_ids.size(0)

            # Calculate accuracy for this batch
            with torch.no_grad():
                correct_pairs += (pos_scores > neg_scores).sum().item()
                total_pairs += q_ids.size(0)

        avg_loss = total_loss / total_pairs if total_pairs > 0 else 0.0
        accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0.0
        return avg_loss, accuracy

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        correct_pairs = 0
        total_pairs = 0

        with torch.no_grad():
            for batch in dataloader:
                q_ids, pos_ids, neg_ids = [t.to(self.device) for t in batch]

                pos_scores = self.model(q_ids, pos_ids)
                neg_scores = self.model(q_ids, neg_ids)

                targets = torch.ones(pos_scores.size(), device=self.device)
                loss = self.criterion(pos_scores, neg_scores, targets)

                total_loss += loss.item() * q_ids.size(0)
                correct_pairs += (pos_scores > neg_scores).sum().item()
                total_pairs += q_ids.size(0)

        avg_loss = total_loss / total_pairs if total_pairs > 0 else 0.0
        accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0.0
        return avg_loss, accuracy

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=Config.NUM_EPOCHS,
        lr=Config.LEARNING_RATE,
    ):
        print(f"Starting Ranker training for {epochs} epochs on {self.device}...")
        optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY
        )

        patience = Config.EARLY_STOPPING_PATIENCE
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer)
            val_loss, val_acc = self.validate(val_loader)

            print(
                f"Epoch {epoch}/{epochs} - "
                f"Train Loss: {train_loss}, Train Acc: {train_acc} - "
                f"Val Loss: {val_loss}, Val Acc: {val_acc}"
            )

            # Checkpoint and Early Stopping
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), Config.RANKER_MODEL_PATH)
                print(f"New best model saved to {Config.RANKER_MODEL_PATH}")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break


class ReaderTrainer:
    """
    Trainer for the Quasi-Recurrent (QRNN) Reader.
    Optimizes using Categorical Cross-Entropy Loss for start and end positions.
    """

    def __init__(self, model, device=Config.DEVICE):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_loss = float("inf")

    def train_epoch(self, dataloader, optimizer):
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in dataloader:
            # Unpack batch (collated by reader_collate_fn)
            input_ids, start_targets, end_targets = [t.to(self.device) for t in batch]

            optimizer.zero_grad()

            # Forward pass
            start_logits, end_logits = self.model(input_ids)

            # Compute loss
            loss_start = self.criterion(start_logits, start_targets)
            loss_end = self.criterion(end_logits, end_targets)
            loss = (loss_start + loss_end) / 2.0

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return avg_loss

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        exact_matches = 0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids, start_targets, end_targets = [
                    t.to(self.device) for t in batch
                ]

                start_logits, end_logits = self.model(input_ids)

                loss_start = self.criterion(start_logits, start_targets)
                loss_end = self.criterion(end_logits, end_targets)
                loss = (loss_start + loss_end) / 2.0

                total_loss += loss.item() * input_ids.size(0)

                # Calculate Exact Match (EM)
                pred_starts = torch.argmax(start_logits, dim=1)
                pred_ends = torch.argmax(end_logits, dim=1)

                # Check if both start and end match targets
                matches = (pred_starts == start_targets) & (pred_ends == end_targets)
                exact_matches += matches.sum().item()

                total_samples += input_ids.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        em_score = exact_matches / total_samples if total_samples > 0 else 0.0
        return avg_loss, em_score

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=Config.NUM_EPOCHS,
        lr=Config.LEARNING_RATE,
    ):
        print(f"Starting Reader training for {epochs} epochs on {self.device}...")
        optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY
        )

        patience = Config.EARLY_STOPPING_PATIENCE
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, optimizer)
            val_loss, val_em = self.validate(val_loader)

            print(
                f"Epoch {epoch}/{epochs} - "
                f"Train Loss: {train_loss} - "
                f"Val Loss: {val_loss}, Val EM: {val_em}"
            )

            # Checkpoint and Early Stopping based on Validation Loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), Config.READER_MODEL_PATH)
                print(f"New best model saved to {Config.READER_MODEL_PATH}")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break
