import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from library.utils import calculate_map, set_seed
from library.losses import MultiTaskLoss
from library.dataset import SaltDataset, get_transforms


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch using the supervised Multi-Task strategy.

    Args:
        model: The PyTorch model (ResNet34WideLinkNet).
        loader: DataLoader for labeled training data.
        optimizer: Optimizer.
        criterion: MultiTaskLoss.
        device: torch.device.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (images, masks, depths, ids) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        depths = depths.to(device)

        optimizer.zero_grad()

        # Forward pass
        mask_logits, depth_preds = model(images)

        # Calculate loss
        loss, loss_dict = criterion(mask_logits, masks, depth_preds, depths)

        # Runtime Assertion: Verify Auxiliary Depth Head Connection
        # 1. Check if depth predictions require gradients (graph connected)
        if not depth_preds.requires_grad:
            raise RuntimeError(
                "Depth predictions do not require gradients! Check model freezing or graph disconnection."
            )

        # 2. Check if depth loss is non-trivial ( > 0).
        # We use a small epsilon to allow for near-zero loss in perfect convergence,
        # but practically it should be positive.
        if loss_dict["loss_depth"] <= 0:
            # In rare cases of perfect prediction it could be 0, but usually this indicates a bug in data/targets
            # We allow 0.0 only if targets equal preds, but here we warn or assert based on instruction.
            # The instruction says "verify ... > 0".
            if loss_dict["loss_depth"] < 0:
                raise RuntimeError(f"Depth loss is negative: {loss_dict['loss_depth']}")

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def train_student_epoch(
    model, labeled_loader, unlabeled_loader, optimizer, criterion, device
):
    """
    Trains the Student model for one epoch using Semi-Supervised Learning.
    Mixes labeled data (Multi-Task Loss) and unlabeled data (BCE with Soft/Hard Pseudo-labels).

    Args:
        model: The Student model.
        labeled_loader: DataLoader for labeled data.
        unlabeled_loader: DataLoader for unlabeled data (with pseudo-labels).
        optimizer: Optimizer.
        criterion: MultiTaskLoss (for labeled data).
        device: torch.device.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    # We iterate over both loaders. If lengths differ, we zip which truncates to the shorter one.
    # Ideally loaders should be balanced or infinite, but for this task we iterate once per epoch definition.

    iter_labeled = iter(labeled_loader)
    iter_unlabeled = iter(unlabeled_loader)

    steps = min(len(labeled_loader), len(unlabeled_loader))

    bce_unlabeled = nn.BCEWithLogitsLoss()

    for _ in range(steps):
        # Load Labeled Batch
        try:
            l_images, l_masks, l_depths, _ = next(iter_labeled)
        except StopIteration:
            iter_labeled = iter(labeled_loader)
            l_images, l_masks, l_depths, _ = next(iter_labeled)

        # Load Unlabeled Batch (Pseudo-labels)
        try:
            u_images, u_masks, u_depths, _ = next(iter_unlabeled)
        except StopIteration:
            iter_unlabeled = iter(unlabeled_loader)
            u_images, u_masks, u_depths, _ = next(iter_unlabeled)

        # Move to device
        l_images = l_images.to(device)
        l_masks = l_masks.to(device)
        l_depths = l_depths.to(device)

        u_images = u_images.to(device)
        u_masks = u_masks.to(device)  # Pseudo-labels (Binary from RLE or Soft)

        optimizer.zero_grad()

        # --- Labeled Step ---
        l_logits, l_depth_preds = model(l_images)
        loss_labeled, _ = criterion(l_logits, l_masks, l_depth_preds, l_depths)

        # --- Unlabeled Step ---
        # Student ignores depth head for unlabeled data (no ground truth depth usually)
        u_logits, _ = model(u_images)

        # Loss is BCE against pseudo-labels
        # u_masks needs to be float for BCEWithLogitsLoss
        loss_unlabeled = bce_unlabeled(u_logits, u_masks.float())

        # Combine losses
        loss = loss_labeled + loss_unlabeled

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / steps


def validate(
    model,
    loader,
    device,
    thresholds=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95),
):
    """
    Evaluates the model on the validation set using Mean Average Precision (mAP).

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        device: torch.device.
        thresholds: Tuple of IoU thresholds.

    Returns:
        float: The mAP score.
    """
    model.eval()
    all_preds = []
    all_gts = []

    with torch.no_grad():
        for images, masks, depths, ids in loader:
            images = images.to(device)

            # Forward pass (ignore depth for validation metric)
            logits, _ = model(images)

            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(logits)

            # Store predictions and ground truths
            # Move to CPU to save GPU memory
            all_preds.append(preds.cpu())
            all_gts.append(masks)

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_gts = torch.cat(all_gts, dim=0)

    # Calculate mAP
    score = calculate_map(all_preds, all_gts, thresholds=thresholds)

    return score


class SaltTrainer:
    """
    Wrapper class to manage training, validation, early stopping, and checkpointing.
    """

    def __init__(
        self,
        model,
        device,
        optimizer,
        scheduler=None,
        criterion=None,
        checkpoint_dir="./working/idea_27",
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion if criterion else MultiTaskLoss()
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_score = -1.0
        self.patience_counter = 0

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=50,
        patience=10,
        student_mode=False,
        unlabeled_loader=None,
    ):
        """
        Runs the training loop.

        Args:
            train_loader: Labeled training data.
            val_loader: Validation data.
            epochs: Max epochs.
            patience: Early stopping patience.
            student_mode: If True, uses train_student_epoch with unlabeled_loader.
            unlabeled_loader: Required if student_mode is True.
        """
        print(f"Starting training (Student Mode: {student_mode})...")

        for epoch in range(1, epochs + 1):
            if student_mode:
                if unlabeled_loader is None:
                    raise ValueError(
                        "unlabeled_loader must be provided for student mode"
                    )
                train_loss = train_student_epoch(
                    self.model,
                    train_loader,
                    unlabeled_loader,
                    self.optimizer,
                    self.criterion,
                    self.device,
                )
            else:
                train_loss = train_epoch(
                    self.model,
                    train_loader,
                    self.optimizer,
                    self.criterion,
                    self.device,
                )

            val_score = validate(self.model, val_loader, self.device)

            # Update Scheduler
            if self.scheduler:
                # Assuming CosineAnnealing or similar that steps per epoch
                self.scheduler.step()

            # Print metrics
            print(
                f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val mAP: {val_score:.10f}"
            )

            # Checkpoint & Early Stopping
            if val_score > self.best_score:
                self.best_score = val_score
                self.patience_counter = 0
                self.save_checkpoint("best_model.pth")
                print(f"  -> New best model saved! (Score: {self.best_score:.10f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(
                        f"Early stopping triggered after {patience} epochs without improvement."
                    )
                    break

    def save_checkpoint(self, filename):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, filename):
        path = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Loaded checkpoint from {path}")
        else:
            print(f"Checkpoint {path} not found.")
