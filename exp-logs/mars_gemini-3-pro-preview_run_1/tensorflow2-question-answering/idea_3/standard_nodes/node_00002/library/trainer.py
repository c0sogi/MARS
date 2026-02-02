import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from library.utils import save_checkpoint, print_metrics


class ModelTrainer:
    """
    Manages the training and validation lifecycle of the DanTqpModel.
    """

    def __init__(self, model, device, learning_rate=0.001):
        """
        Args:
            model (nn.Module): The DanTqpModel to train.
            device (torch.device): Device to run training on.
            learning_rate (float): Learning rate for the optimizer.
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Loss functions
        self.ranker_criterion = nn.BCEWithLogitsLoss()
        # ignore_index=-1 handles padding from collate_fn
        self.extractor_criterion = nn.CrossEntropyLoss(
            ignore_index=-1, reduction="none"
        )

    def train_epoch(self, dataloader, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        running_ranker_loss = 0.0
        running_extractor_loss = 0.0

        all_long_preds = []
        all_long_labels = []

        for batch_idx, batch in enumerate(dataloader):
            # Move data to device
            q_input_ids = batch["q_input_ids"].to(self.device)
            c_input_ids = batch["c_input_ids"].to(self.device)
            label_long = batch["label_long"].to(self.device)  # (batch_size,)
            sa_labels = batch["sa_labels"].to(self.device)  # (batch_size, seq_len)

            # Forward pass
            self.optimizer.zero_grad()
            ranker_logits, extractor_logits = self.model(q_input_ids, c_input_ids)

            # 1. Ranker Loss (Binary Classification)
            # ranker_logits: (batch, 1) -> squeeze to (batch,)
            loss_ranker = self.ranker_criterion(ranker_logits.squeeze(-1), label_long)

            # 2. Extractor Loss (Token Classification)
            # extractor_logits: (batch, seq_len, 3) -> reshape to (batch*seq_len, 3)
            # sa_labels: (batch, seq_len) -> reshape to (batch*seq_len)
            batch_size, seq_len, num_classes = extractor_logits.shape
            flat_logits = extractor_logits.view(-1, num_classes)
            flat_labels = sa_labels.view(-1)

            raw_extractor_loss = self.extractor_criterion(flat_logits, flat_labels)

            # Reshape loss back to (batch, seq_len) to apply mask
            raw_extractor_loss = raw_extractor_loss.view(batch_size, seq_len)

            # Mask: Only penalize extractor for positive long answer samples
            # label_long is 1.0 for positive, 0.0 for negative
            # Expand label_long to (batch, seq_len)
            mask = label_long.unsqueeze(1).expand_as(raw_extractor_loss)

            # Compute masked mean
            # We divide by the number of active tokens in positive samples + epsilon
            # Note: ignore_index in CrossEntropyLoss handles padding, so raw_extractor_loss is 0 at pads
            masked_loss = raw_extractor_loss * mask
            num_active_elements = (
                mask.sum() * seq_len
            )  # Approximation, or sum of non-pad mask
            # Better normalization: sum of mask where labels != -1
            valid_tokens = (sa_labels != -1).float()
            normalization_factor = (mask * valid_tokens).sum().clamp(min=1.0)

            loss_extractor = masked_loss.sum() / normalization_factor

            # Total Loss
            total_loss = loss_ranker + loss_extractor

            # Backward
            total_loss.backward()
            self.optimizer.step()

            # Metrics Tracking
            running_loss += total_loss.item()
            running_ranker_loss += loss_ranker.item()
            running_extractor_loss += loss_extractor.item()

            preds = torch.sigmoid(ranker_logits.squeeze(-1)) > 0.5
            all_long_preds.extend(preds.cpu().numpy())
            all_long_labels.extend(label_long.cpu().numpy())

        avg_loss = running_loss / len(dataloader)
        avg_ranker = running_ranker_loss / len(dataloader)
        avg_extractor = running_extractor_loss / len(dataloader)

        long_acc = accuracy_score(all_long_labels, all_long_preds)
        long_f1 = f1_score(all_long_labels, all_long_preds, zero_division=0)

        metrics = {
            "Train Loss": avg_loss,
            "Train Ranker Loss": avg_ranker,
            "Train Extractor Loss": avg_extractor,
            "Train Long Acc": long_acc,
            "Train Long F1": long_f1,
        }
        print_metrics(metrics, prefix=f"Epoch {epoch_idx}")
        return avg_loss

    def validate(self, dataloader):
        """
        Runs validation loop.
        """
        self.model.eval()
        running_loss = 0.0

        all_long_preds = []
        all_long_labels = []

        # For Short Answer metrics (Token level accuracy on positive samples)
        sa_correct = 0
        sa_total = 0

        with torch.no_grad():
            for batch in dataloader:
                q_input_ids = batch["q_input_ids"].to(self.device)
                c_input_ids = batch["c_input_ids"].to(self.device)
                label_long = batch["label_long"].to(self.device)
                sa_labels = batch["sa_labels"].to(self.device)

                ranker_logits, extractor_logits = self.model(q_input_ids, c_input_ids)

                # Ranker Loss
                loss_ranker = self.ranker_criterion(
                    ranker_logits.squeeze(-1), label_long
                )

                # Extractor Loss (Masked)
                batch_size, seq_len, num_classes = extractor_logits.shape
                flat_logits = extractor_logits.view(-1, num_classes)
                flat_labels = sa_labels.view(-1)
                raw_extractor_loss = self.extractor_criterion(flat_logits, flat_labels)
                raw_extractor_loss = raw_extractor_loss.view(batch_size, seq_len)
                mask = label_long.unsqueeze(1).expand_as(raw_extractor_loss)
                valid_tokens = (sa_labels != -1).float()
                normalization_factor = (mask * valid_tokens).sum().clamp(min=1.0)
                loss_extractor = (
                    raw_extractor_loss * mask
                ).sum() / normalization_factor

                total_loss = loss_ranker + loss_extractor
                running_loss += total_loss.item()

                # Long Answer Metrics
                preds = torch.sigmoid(ranker_logits.squeeze(-1)) > 0.5
                all_long_preds.extend(preds.cpu().numpy())
                all_long_labels.extend(label_long.cpu().numpy())

                # Short Answer Metrics (Only for positive samples)
                # Get indices where label_long is 1
                pos_indices = (label_long == 1).nonzero(as_tuple=True)[0]
                if len(pos_indices) > 0:
                    pos_logits = extractor_logits[pos_indices]  # (num_pos, seq_len, 3)
                    pos_labels = sa_labels[pos_indices]  # (num_pos, seq_len)

                    pos_preds = torch.argmax(pos_logits, dim=2)  # (num_pos, seq_len)

                    # Mask padding (-1)
                    valid_mask = pos_labels != -1

                    correct = (pos_preds == pos_labels) & valid_mask
                    sa_correct += correct.sum().item()
                    sa_total += valid_mask.sum().item()

        avg_loss = running_loss / len(dataloader)
        long_acc = accuracy_score(all_long_labels, all_long_preds)
        long_f1 = f1_score(all_long_labels, all_long_preds, zero_division=0)
        long_prec = precision_score(all_long_labels, all_long_preds, zero_division=0)
        long_rec = recall_score(all_long_labels, all_long_preds, zero_division=0)

        sa_acc = sa_correct / sa_total if sa_total > 0 else 0.0

        metrics = {
            "Val Loss": avg_loss,
            "Val Long Acc": long_acc,
            "Val Long F1": long_f1,
            "Val Long P": long_prec,
            "Val Long R": long_rec,
            "Val Short Token Acc": sa_acc,
        }
        print_metrics(metrics, prefix="Val")

        # We optimize on Loss, but return F1 for monitoring
        return avg_loss, long_f1

    def train(
        self,
        train_loader,
        val_loader,
        epochs=5,
        patience=3,
        save_path="./working/idea_3/best_model.pth",
    ):
        """
        Executes the full training workflow with early stopping.
        """
        print("Starting training...")
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.train_epoch(train_loader, epoch)
            val_loss, val_f1 = self.validate(val_loader)

            # Early Stopping Check based on Validation Loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(self.model, self.optimizer, epoch, val_loss, save_path)
                print(f"New best model saved with Val Loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print("Training complete.")
