import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from library.config import Config


def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(Config.DEVICE)
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss Functions
        self.rank_criterion = nn.BCEWithLogitsLoss()
        self.span_criterion = nn.CrossEntropyLoss(
            reduction="none"
        )  # 'none' to allow masking
        self.yn_criterion = nn.CrossEntropyLoss()

    def calculate_loss(self, outputs, batch):
        # Unpack targets
        rank_labels = batch["rank_label"].to(self.device)  # (B,)
        span_start = batch["span_start"].to(self.device)  # (B,)
        span_end = batch["span_end"].to(self.device)  # (B,)
        yn_labels = batch["yn_label"].to(self.device)  # (B,)

        # Unpack outputs
        rank_logits = outputs["rank_logits"]
        start_logits = outputs["start_logits"]
        end_logits = outputs["end_logits"]
        yn_logits = outputs["yn_logits"]

        # 1. Ranking Loss (Binary)
        loss_rank = self.rank_criterion(rank_logits, rank_labels)

        # 2. Span Loss (Categorical)
        # Only compute span loss for positive samples (rank_label == 1)
        # We assume rank_label is 0.0 or 1.0.
        loss_start = self.span_criterion(start_logits, span_start)
        loss_end = self.span_criterion(end_logits, span_end)

        # Masking: multiply by rank_label so negatives don't contribute to span loss
        # Note: rank_labels is float, losses are float.
        masked_span_loss = (loss_start * rank_labels) + (loss_end * rank_labels)

        # Normalize by number of positive samples to avoid scaling issues
        num_positives = torch.sum(rank_labels)
        if num_positives > 0:
            loss_span = masked_span_loss.sum() / num_positives
        else:
            loss_span = torch.tensor(0.0, device=self.device)

        # 3. Yes/No Loss (Categorical)
        # We train Y/N on all samples. Negatives are labeled 'NONE' (class 0).
        loss_yn = self.yn_criterion(yn_logits, yn_labels)

        # Weighted Sum
        total_loss = (
            Config.LOSS_WEIGHT_RANK * loss_rank
            + Config.LOSS_WEIGHT_SPAN * loss_span
            + Config.LOSS_WEIGHT_YN * loss_yn
        )

        return total_loss, loss_rank.item(), loss_span.item(), loss_yn.item()

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0

        for batch in self.train_loader:
            # Move inputs to device
            q_ids = batch["q_ids"].to(self.device)
            c_ids = batch["c_ids"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(q_ids, c_ids)

            # Calculate loss
            loss, _, _, _ = self.calculate_loss(outputs, batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                q_ids = batch["q_ids"].to(self.device)
                c_ids = batch["c_ids"].to(self.device)

                outputs = self.model(q_ids, c_ids)
                loss, _, _, _ = self.calculate_loss(outputs, batch)

                running_loss += loss.item()

        return running_loss / len(self.val_loader)

    def train(self, epochs=Config.NUM_EPOCHS):
        set_seed(Config.SEED)
        print(f"Starting training on device: {self.device}")

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.evaluate()

            print(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}"
            )

            # Early Stopping and Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                # print(f"New best model saved to {Config.MODEL_SAVE_PATH}")
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

        print("Training complete.")
