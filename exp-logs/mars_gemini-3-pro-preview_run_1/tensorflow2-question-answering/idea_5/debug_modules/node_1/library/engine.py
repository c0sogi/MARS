import torch
import torch.nn as nn
import numpy as np
from library.config import Config


class Engine:
    def __init__(self, model, device, optimizer=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer

        # Loss functions
        self.criterion_long = nn.BCELoss()
        # ignore_index=-1 handles cases where short_start/short_end are -1 (no answer/truncated)
        self.criterion_short = nn.CrossEntropyLoss(ignore_index=-1)

    def train_one_epoch(self, data_loader):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_long_loss = 0.0
        total_short_loss = 0.0

        # Metrics
        correct_long = 0
        total_samples = 0

        for batch_idx, batch in enumerate(data_loader):
            # Move data to device
            q_ids = batch["question_ids"].to(self.device)
            c_ids = batch["candidate_ids"].to(self.device)
            long_labels = (
                batch["long_labels"].to(self.device).float().unsqueeze(1)
            )  # (batch, 1)
            start_labels = batch["short_start_labels"].to(self.device)
            end_labels = batch["short_end_labels"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            long_prob, start_logits, end_logits = self.model(q_ids, c_ids)

            # Calculate Losses
            loss_long = self.criterion_long(long_prob, long_labels)
            loss_start = self.criterion_short(start_logits, start_labels)
            loss_end = self.criterion_short(end_logits, end_labels)

            # Combine losses
            # We weight them equally here, but this could be tuned
            loss = loss_long + loss_start + loss_end

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            batch_size = q_ids.size(0)
            total_loss += loss.item() * batch_size
            total_long_loss += loss_long.item() * batch_size
            total_short_loss += (loss_start.item() + loss_end.item()) * batch_size

            # Long answer accuracy (threshold 0.5 for training metric)
            preds_long = (long_prob > 0.5).float()
            correct_long += (preds_long == long_labels).sum().item()
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_long_loss = total_long_loss / total_samples
        avg_short_loss = total_short_loss / total_samples
        acc_long = correct_long / total_samples

        return {
            "loss": avg_loss,
            "long_loss": avg_long_loss,
            "short_loss": avg_short_loss,
            "long_acc": acc_long,
        }

    def evaluate(self, data_loader):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        total_loss = 0.0

        correct_long = 0
        correct_span = 0
        total_samples = 0
        total_short_samples = 0  # Only count samples where a short answer actually exists for span accuracy

        with torch.no_grad():
            for batch in data_loader:
                q_ids = batch["question_ids"].to(self.device)
                c_ids = batch["candidate_ids"].to(self.device)
                long_labels = batch["long_labels"].to(self.device).float().unsqueeze(1)
                start_labels = batch["short_start_labels"].to(self.device)
                end_labels = batch["short_end_labels"].to(self.device)

                # Forward pass
                long_prob, start_logits, end_logits = self.model(q_ids, c_ids)

                # Calculate Losses
                loss_long = self.criterion_long(long_prob, long_labels)
                loss_start = self.criterion_short(start_logits, start_labels)
                loss_end = self.criterion_short(end_logits, end_labels)
                loss = loss_long + loss_start + loss_end

                batch_size = q_ids.size(0)
                total_loss += loss.item() * batch_size

                # Long Answer Accuracy
                # Using configured threshold for consistency, though 0.5 is standard for binary metrics
                preds_long = (long_prob > Config.LONG_ANSWER_THRESHOLD).float()
                correct_long += (preds_long == long_labels).sum().item()

                # Short Answer Exact Match
                # Only evaluate on samples that have a valid short answer (label != -1)
                # We assume if start is valid, end is valid based on data loader logic
                valid_mask = start_labels != -1
                if valid_mask.sum() > 0:
                    pred_start = torch.argmax(start_logits, dim=1)
                    pred_end = torch.argmax(end_logits, dim=1)

                    # Check exact match on valid samples
                    start_match = pred_start[valid_mask] == start_labels[valid_mask]
                    end_match = pred_end[valid_mask] == end_labels[valid_mask]
                    span_match = start_match & end_match

                    correct_span += span_match.sum().item()
                    total_short_samples += valid_mask.sum().item()

                total_samples += batch_size

        avg_loss = total_loss / total_samples
        acc_long = correct_long / total_samples
        acc_span = (
            correct_span / total_short_samples if total_short_samples > 0 else 0.0
        )

        return {"loss": avg_loss, "long_acc": acc_long, "span_acc": acc_span}

    def fit(self, train_loader, val_loader, epochs, patience, save_path):
        """
        Runs the full training loop with early stopping.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"[Engine] Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Train
            train_metrics = self.train_one_epoch(train_loader)
            print(f"Train Loss: {train_metrics['loss']}")
            print(f"Train Long Acc: {train_metrics['long_acc']}")

            # Validate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                print(f"Val Loss: {val_metrics['loss']}")
                print(f"Val Long Acc: {val_metrics['long_acc']}")
                print(f"Val Span Acc: {val_metrics['span_acc']}")

                # Early Stopping Check
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    patience_counter = 0
                    print(f"Validation loss improved. Saving model to {save_path}")
                    torch.save(self.model.state_dict(), save_path)
                else:
                    patience_counter += 1
                    print(
                        f"Validation loss did not improve. Patience: {patience_counter}/{patience}"
                    )

                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break
            else:
                # If no validation set, just save every epoch
                print(f"Saving model to {save_path}")
                torch.save(self.model.state_dict(), save_path)
