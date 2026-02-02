import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.utils import calculate_f1_score


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where:
        p_t = p if y=1 else 1-p
        alpha_t = alpha if y=1 else 1-alpha
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits (batch_size, num_classes)
        # targets: binary labels (batch_size, num_classes)

        # Compute BCE with logits
        # reduction='none' ensures we get a loss per element to apply weighting
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Calculate p_t
        # bce_loss = -ln(p_t) => p_t = exp(-bce_loss)
        pt = torch.exp(-bce_loss)

        # Calculate alpha_t
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Apply Focal Loss formula
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class Trainer:
    """
    Generic trainer class capable of handling both Wide (Linear) and Deep (Transformer) models.
    """

    def __init__(
        self, model, optimizer, criterion, device, scheduler=None, save_path=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.save_path = save_path
        self.best_score = -float("inf")

    def train_epoch(self, dataloader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for batch in dataloader:
            # Move all tensors to device
            batch = [t.to(self.device) for t in batch]

            # Dispatch based on batch structure
            # WideDataset returns (x, y)
            # DeepDataset returns (input_ids, mask, y)

            if len(batch) == 2:
                # Could be Wide Train (x, y) or Deep Test (ids, mask)
                # Check dtype: Wide features are Float, Deep IDs are Long
                if batch[0].dtype == torch.long:
                    raise ValueError(
                        "Received batch with 2 elements (input_ids, mask) but expected targets for training."
                    )

                # Wide Model Training
                inputs, targets = batch
                outputs = self.model(inputs)

            elif len(batch) == 3:
                # Deep Model Training
                input_ids, mask, targets = batch
                outputs = self.model(input_ids, attention_mask=mask)

            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")

            # Compute loss
            loss = self.criterion(outputs, targets)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Step scheduler if per-step
            if self.scheduler:
                self.scheduler.step()

            running_loss += loss.item() * targets.size(0)
            count += targets.size(0)

        return running_loss / count if count > 0 else 0.0

    def validate(self, dataloader):
        """
        Runs validation and computes F1 score.
        """
        self.model.eval()
        running_loss = 0.0
        count = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                batch = [t.to(self.device) for t in batch]

                if len(batch) == 2:
                    # Wide Model
                    inputs, targets = batch
                    outputs = self.model(inputs)
                elif len(batch) == 3:
                    # Deep Model
                    input_ids, mask, targets = batch
                    outputs = self.model(input_ids, attention_mask=mask)
                else:
                    raise ValueError(f"Unexpected batch length: {len(batch)}")

                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * targets.size(0)
                count += targets.size(0)

                # Store predictions for F1 calculation
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(outputs)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        epoch_loss = running_loss / count if count > 0 else 0.0

        val_f1 = 0.0
        if len(all_preds) > 0:
            all_preds = np.vstack(all_preds)
            all_targets = np.vstack(all_targets)

            # Use 0.5 threshold for validation monitoring
            y_pred_bin = (all_preds >= 0.5).astype(int)
            val_f1 = calculate_f1_score(all_targets, y_pred_bin, average="samples")

        return epoch_loss, val_f1

    def fit(self, train_loader, val_loader, epochs, patience=3):
        """
        Runs the full training loop with Early Stopping.
        """
        print(f"Starting training on {self.device} for {epochs} epochs.")

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_f1 = self.validate(val_loader)

            # Print full precision metrics
            print(
                f"Epoch {epoch}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val F1: {val_f1}"
            )

            # Checkpoint and Early Stopping
            if val_f1 > self.best_score:
                self.best_score = val_f1
                patience_counter = 0
                if self.save_path:
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                    torch.save(self.model.state_dict(), self.save_path)
                    print(f"New best model saved to {self.save_path}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs with no improvement."
                )
                break

        # Load best model state
        if self.save_path and os.path.exists(self.save_path):
            print(f"Loading best model from {self.save_path}")
            self.model.load_state_dict(
                torch.load(self.save_path, map_location=self.device)
            )

        return self.best_score


def predict_logits(model, dataloader, device):
    """
    Generates raw logits for a dataset.
    Handles input differences between Wide and Deep models automatically.

    Returns:
        np.ndarray: Logits matrix of shape (n_samples, n_classes)
    """
    model.eval()
    model.to(device)
    all_logits = []

    with torch.no_grad():
        for batch in dataloader:
            batch = [t.to(device) for t in batch]

            # Determine input type
            # Wide Test: (x,) -> len 1
            # Wide Train/Val (ignoring targets): (x, y) -> len 2, x is float
            # Deep Test: (ids, mask) -> len 2, ids is long
            # Deep Train/Val (ignoring targets): (ids, mask, y) -> len 3

            if len(batch) == 1:
                # Wide Test
                inputs = batch[0]
                outputs = model(inputs)
            elif len(batch) == 2:
                if batch[0].dtype == torch.long:
                    # Deep Test
                    input_ids, mask = batch
                    outputs = model(input_ids, attention_mask=mask)
                else:
                    # Wide Train/Val
                    inputs, _ = batch
                    outputs = model(inputs)
            elif len(batch) == 3:
                # Deep Train/Val
                input_ids, mask, _ = batch
                outputs = model(input_ids, attention_mask=mask)
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")

            all_logits.append(outputs.cpu().numpy())

    if len(all_logits) > 0:
        return np.vstack(all_logits)
    return np.array([])
