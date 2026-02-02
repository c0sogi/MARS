import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import save_checkpoint


class Trainer:
    """
    Manages the training and validation lifecycle of the Feed-Forward Decomposable Attention Network.
    """

    def __init__(self, model, train_loader, val_loader, config: Config):
        """
        Args:
            model (nn.Module): The neural network model.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            config (Config): Configuration object.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(config.DEVICE)

        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)

        # Loss Functions
        # Ranking: Binary classification (Is this candidate the answer?)
        self.ranking_criterion = nn.BCEWithLogitsLoss()

        # Span: Multi-class classification over sequence length (Which token is start/end?)
        self.span_criterion = nn.CrossEntropyLoss()

        # Yes/No: Multi-class classification (YES, NO, NONE)
        self.yn_criterion = nn.CrossEntropyLoss()

    def compute_loss(self, outputs, batch):
        """
        Computes the multi-task loss.

        Args:
            outputs (dict): Model outputs containing logits.
            batch (dict): Batch data containing targets.

        Returns:
            tuple: (total_loss, metrics_dict)
        """
        # Unpack targets
        long_labels = batch["long_labels"].to(self.device)  # (B,) Float
        short_starts = batch["short_starts"].to(self.device)  # (B,) Long
        short_ends = batch["short_ends"].to(self.device)  # (B,) Long
        yn_labels = batch["yn_labels"].to(self.device)  # (B,) Long

        # Unpack outputs
        ranking_logits = outputs["ranking_logits"].squeeze(-1)  # (B,)
        start_logits = outputs["start_logits"]  # (B, C_Len)
        end_logits = outputs["end_logits"]  # (B, C_Len)
        yn_logits = outputs["yn_logits"]  # (B, 3)

        # 1. Ranking Loss (Binary Cross Entropy)
        ranking_loss = self.ranking_criterion(ranking_logits, long_labels)

        # 2. Span Loss (Cross Entropy)
        # Only compute span loss for positive candidates (long_labels == 1)
        # Negative candidates have dummy span targets (0,0) which we shouldn't learn.
        pos_mask = long_labels == 1.0

        if pos_mask.sum() > 0:
            start_loss = self.span_criterion(
                start_logits[pos_mask], short_starts[pos_mask]
            )
            end_loss = self.span_criterion(end_logits[pos_mask], short_ends[pos_mask])
        else:
            start_loss = torch.tensor(0.0, device=self.device)
            end_loss = torch.tensor(0.0, device=self.device)

        # 3. Yes/No Loss (Cross Entropy)
        # We train this on all candidates. Negative candidates are labeled NONE (2).
        yn_loss = self.yn_criterion(yn_logits, yn_labels)

        # Total Loss
        total_loss = ranking_loss + start_loss + end_loss + yn_loss

        # Metrics Calculation (for monitoring)
        with torch.no_grad():
            # Ranking Accuracy
            preds = (torch.sigmoid(ranking_logits) > 0.5).float()
            rank_acc = (preds == long_labels).float().mean()

            # Yes/No Accuracy
            yn_preds = torch.argmax(yn_logits, dim=1)
            yn_acc = (yn_preds == yn_labels).float().mean()

            # Span Accuracy (Exact Match, only on positives)
            if pos_mask.sum() > 0:
                start_preds = torch.argmax(start_logits[pos_mask], dim=1)
                end_preds = torch.argmax(end_logits[pos_mask], dim=1)
                span_acc = (
                    (
                        (start_preds == short_starts[pos_mask])
                        & (end_preds == short_ends[pos_mask])
                    )
                    .float()
                    .mean()
                )
            else:
                span_acc = torch.tensor(0.0, device=self.device)

        metrics = {
            "loss": total_loss.item(),
            "rank_acc": rank_acc.item(),
            "span_acc": span_acc.item(),
            "yn_acc": yn_acc.item(),
        }

        return total_loss, metrics

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_metrics = {"loss": 0.0, "rank_acc": 0.0, "span_acc": 0.0, "yn_acc": 0.0}
        num_batches = 0

        for batch in self.train_loader:
            # Move inputs to device
            q_input = batch["q_input"].to(self.device)
            c_input = batch["c_input"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(q_input, c_input)

            # Compute loss
            loss, metrics = self.compute_loss(outputs, batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            for k, v in metrics.items():
                running_metrics[k] += v
            num_batches += 1

        # Average metrics
        avg_metrics = {k: v / num_batches for k, v in running_metrics.items()}
        return avg_metrics

    def validate(self):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        running_metrics = {"loss": 0.0, "rank_acc": 0.0, "span_acc": 0.0, "yn_acc": 0.0}
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                q_input = batch["q_input"].to(self.device)
                c_input = batch["c_input"].to(self.device)

                outputs = self.model(q_input, c_input)
                loss, metrics = self.compute_loss(outputs, batch)

                for k, v in metrics.items():
                    running_metrics[k] += v
                num_batches += 1

        avg_metrics = {k: v / num_batches for k, v in running_metrics.items()}
        return avg_metrics

    def train(self):
        """
        Main training loop with early stopping.
        """
        print(f"Starting training on device: {self.device}")

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            print(f"Epoch {epoch}/{self.config.NUM_EPOCHS}")

            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"  Train Loss: {train_metrics['loss']}")
            print(f"  Train Rank Acc: {train_metrics['rank_acc']}")
            print(f"  Train Span Acc: {train_metrics['span_acc']}")
            print(f"  Train Y/N Acc: {train_metrics['yn_acc']}")

            # Validate
            val_metrics = self.validate()
            val_loss = val_metrics["loss"]
            print(f"  Val Loss: {val_loss}")
            print(f"  Val Rank Acc: {val_metrics['rank_acc']}")
            print(f"  Val Span Acc: {val_metrics['span_acc']}")
            print(f"  Val Y/N Acc: {val_metrics['yn_acc']}")

            # Checkpoint and Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_loss,
                    self.config.MODEL_CHECKPOINT_PATH,
                )
                print("  New best model saved.")
            else:
                patience_counter += 1
                print(
                    f"  No improvement. Patience: {patience_counter}/{self.config.EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break
