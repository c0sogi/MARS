import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import (
    AverageMeter,
    save_checkpoint,
    mixup_data,
    mixup_criterion,
    calculate_auc,
)


class Trainer:
    """
    Trainer class for the Right Whale Detection task.
    Handles training, validation, and prediction loops.
    """

    def __init__(
        self, model, optimizer, scheduler=None, device=Config.DEVICE, pos_weight=None
    ):
        """
        Args:
            model (nn.Module): The PyTorch model to train.
            optimizer (torch.optim.Optimizer): Optimizer.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
            device (str): Device to run on ('cuda' or 'cpu').
            pos_weight (float, optional): Weight for the positive class in BCE loss.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # Initialize Loss Function
        # Use BCEWithLogitsLoss for numerical stability with logits
        if pos_weight is not None:
            # pos_weight must be a tensor
            weight_tensor = torch.tensor([pos_weight], device=device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

    def train_one_epoch(self, train_loader, epoch, use_mixup=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        losses = AverageMeter()

        for i, (images, targets) in enumerate(train_loader):
            images = images.to(self.device)
            # Ensure targets are (Batch, 1)
            targets = targets.to(self.device).view(-1, 1)

            # Apply Mixup if enabled and alpha is set
            if use_mixup and Config.MIXUP_ALPHA > 0:
                images, targets_a, targets_b, lam = mixup_data(
                    images, targets, Config.MIXUP_ALPHA, self.device
                )

                # Forward pass
                outputs = self.model(images)

                # Calculate Mixup Loss
                loss = mixup_criterion(
                    self.criterion, outputs, targets_a, targets_b, lam
                )
            else:
                # Standard Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            losses.update(loss.item(), images.size(0))

        return losses.avg

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        Returns:
            avg_loss (float): Average validation loss.
            auc (float): Area Under the ROC Curve.
        """
        self.model.eval()
        losses = AverageMeter()

        all_targets = []
        all_preds = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device).view(-1, 1)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                losses.update(loss.item(), images.size(0))

                # Apply sigmoid to get probabilities for AUC calculation
                probs = torch.sigmoid(outputs)

                all_preds.extend(probs.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())

        # Calculate AUC
        auc = calculate_auc(all_targets, all_preds)

        return losses.avg, auc

    def fit(
        self,
        train_loader,
        val_loader,
        epochs,
        patience=5,
        use_mixup=False,
        save_name="best_model.pth",
    ):
        """
        Runs the full training loop with Early Stopping.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            epochs (int): Maximum number of epochs.
            patience (int): Early stopping patience.
            use_mixup (bool): Whether to use Mixup augmentation.
            save_name (str): Filename to save the best model (e.g., 'teacher_best.pth').

        Returns:
            float: Best Validation AUC achieved.
        """
        best_auc = 0.0
        patience_counter = 0

        print(
            f"Starting training for {epochs} epochs. Mixup: {use_mixup}. Patience: {patience}."
        )

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(train_loader, epoch, use_mixup)

            # Validate
            val_loss, val_auc = self.validate(val_loader)

            # Scheduler Step
            if self.scheduler:
                self.scheduler.step()

            duration = time.time() - start_time

            # Print metrics (Full precision as requested)
            print(
                f"Epoch {epoch}: Train Loss {train_loss}, Val Loss {val_loss}, Val AUC {val_auc}, Time {duration}s"
            )

            # Checkpoint and Early Stopping Logic
            is_best = val_auc > best_auc
            if is_best:
                best_auc = val_auc
                patience_counter = 0

                # Save as the generic best model (for immediate reloading if needed)
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_auc,
                    is_best=True,
                )

                # Also save with the specific name provided (e.g., teacher_best.pth)
                # We pass is_best=False here to avoid overwriting 'best_model.pth' again unnecessarily,
                # but we want to ensure this specific file contains the best weights.
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_auc,
                    is_best=False,
                    filename=save_name,
                )
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch}. Best AUC: {best_auc}"
                )
                break

        return best_auc

    def predict(self, test_loader):
        """
        Generates predictions for the test set.

        Returns:
            dict: Mapping of clip_name -> probability
        """
        self.model.eval()
        predictions = {}

        with torch.no_grad():
            for images, clip_names in test_loader:
                images = images.to(self.device)

                outputs = self.model(images)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()

                for name, prob in zip(clip_names, probs):
                    predictions[name] = prob

        return predictions


def get_pos_weight(dataset):
    """
    Calculates the Inverse Class Frequency Weight (num_neg / num_pos).
    Useful for handling class imbalance in BCEWithLogitsLoss.

    Args:
        dataset (WhaleDataset): The dataset containing targets.

    Returns:
        float: The calculated positive weight.
    """
    targets = np.array(dataset.targets)

    # Calculate sums. Works for both hard labels (0/1) and soft labels (probabilities).
    pos_sum = np.sum(targets)
    total = len(targets)
    neg_sum = total - pos_sum

    if pos_sum == 0:
        return 1.0

    return neg_sum / pos_sum


def generate_submission(predictions, output_path):
    """
    Saves predictions to a CSV file in the required format.

    Args:
        predictions (dict): Dictionary of clip_name -> probability.
        output_path (str): Path to save the CSV.
    """
    # Create DataFrame
    df = pd.DataFrame(list(predictions.items()), columns=["clip", "probability"])

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
