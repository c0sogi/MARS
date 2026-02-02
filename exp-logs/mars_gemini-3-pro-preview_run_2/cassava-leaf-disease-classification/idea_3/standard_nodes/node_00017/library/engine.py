import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import CFG
from library.utils import AverageMeter, save_checkpoint


class EarlyStopping:
    """
    Early stops the training if validation accuracy doesn't improve after a given patience.
    """

    def __init__(self, patience=3, mode="max", delta=0.0, save_path=None):
        """
        Args:
            patience (int): How long to wait after last time validation metric improved.
            mode (str): 'min' for loss, 'max' for accuracy.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            save_path (str): Path to save the best checkpoint.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode
        self.delta = delta
        self.save_path = save_path

        if mode == "min":
            self.val_score = np.inf
        else:
            self.val_score = -np.inf

    def __call__(self, score, model, optimizer, epoch):
        if self.mode == "min":
            improved = (
                score < (self.best_score - self.delta)
                if self.best_score is not None
                else True
            )
        else:
            improved = (
                score > (self.best_score + self.delta)
                if self.best_score is not None
                else True
            )

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, optimizer, epoch)
        elif improved:
            self.best_score = score
            self.save_checkpoint(score, model, optimizer, epoch)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, score, model, optimizer, epoch):
        """Saves model when validation metric decreases/increases."""
        if self.save_path:
            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_score": score,
            }
            save_checkpoint(state, self.save_path)


def train_one_epoch(
    epoch, model, train_loader, criterion, optimizer, device, scheduler=None
):
    """
    Trains the model for one epoch.
    """
    model.train()

    losses = AverageMeter()
    scores = AverageMeter()

    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        batch_size = labels.size(0)

        # Forward pass
        y_preds = model(images)
        loss = criterion(y_preds, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update scheduler if it's a per-step scheduler
        if scheduler is not None:
            scheduler.step()

        # Metrics
        losses.update(loss.item(), batch_size)

        # Calculate accuracy
        preds = torch.argmax(y_preds, dim=1)
        acc = (preds == labels).float().mean()
        scores.update(acc.item(), batch_size)

    return losses.avg, scores.avg


def validate(model, val_loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()

    losses = AverageMeter()
    scores = AverageMeter()

    with torch.no_grad():
        for step, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            batch_size = labels.size(0)

            y_preds = model(images)
            loss = criterion(y_preds, labels)

            losses.update(loss.item(), batch_size)

            preds = torch.argmax(y_preds, dim=1)
            acc = (preds == labels).float().mean()
            scores.update(acc.item(), batch_size)

    return losses.avg, scores.avg


def inference_fn(model, test_loader, device):
    """
    Runs inference on the test set, optionally using Test Time Augmentation (TTA).
    Returns a list of predicted labels.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            # Forward pass (Original)
            output = model(images)

            if CFG.tta:
                # Forward pass (Horizontal Flip)
                # dim=3 is width for (N, C, H, W)
                images_flipped = torch.flip(images, dims=[3])
                output_flipped = model(images_flipped)

                # Average logits
                output = (output + output_flipped) / 2.0

            # Get predicted class
            # We use argmax to get the class ID
            batch_preds = torch.argmax(output, dim=1)
            preds.append(batch_preds.cpu().numpy())

    return np.concatenate(preds)


def generate_submission(model, test_loader, device, output_dir="./submission"):
    """
    Generates the submission file.

    Args:
        model: The trained model.
        test_loader: DataLoader for the test set.
        device: Computation device.
        output_dir: Directory to save the submission.csv.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Run inference
    predictions = inference_fn(model, test_loader, device)

    # Retrieve image IDs from the dataset
    # The dataset is expected to be accessible via the loader
    dataset = test_loader.dataset
    image_ids = dataset.df["image_id"].values

    # Create DataFrame
    submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})

    # Save to CSV
    submission_path = os.path.join(output_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    # Print confirmation (optional, but helpful for logs)
    print(f"Submission saved to {submission_path}")
    print(submission_df.head())
