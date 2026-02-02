import os
import time
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from library.utils import AverageMeter
from library.config import Config


def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch, config):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: The training DataLoader.
        optimizer: The optimizer.
        loss_fn: The HierarchicalLoss function.
        device: The device to train on.
        epoch: Current epoch number.
        config: Configuration object.

    Returns:
        dict: Average metrics for the epoch.
    """
    model.train()

    meters = {
        "loss_total": AverageMeter(),
        "loss_species": AverageMeter(),
        "loss_genus": AverageMeter(),
        "loss_family": AverageMeter(),
        "acc_species": AverageMeter(),
    }

    start_time = time.time()

    for batch_idx, (
        images,
        species_targets,
        genus_targets,
        family_targets,
    ) in enumerate(loader):
        # Move data to device
        images = images.to(device, non_blocking=True)
        species_targets = species_targets.to(device, non_blocking=True)
        genus_targets = genus_targets.to(device, non_blocking=True)
        family_targets = family_targets.to(device, non_blocking=True)

        targets = (species_targets, genus_targets, family_targets)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)  # (species_logits, genus_logits, family_logits)

        # Loss calculation
        loss, metrics = loss_fn(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update meters
        batch_size = images.size(0)
        meters["loss_total"].update(metrics["loss_total"], batch_size)
        meters["loss_species"].update(metrics["loss_species"], batch_size)
        meters["loss_genus"].update(metrics["loss_genus"], batch_size)
        meters["loss_family"].update(metrics["loss_family"], batch_size)

        # Calculate simple accuracy for species (for monitoring)
        species_logits = outputs[0]
        preds = torch.argmax(species_logits, dim=1)
        acc = (preds == species_targets).float().mean().item()
        meters["acc_species"].update(acc, batch_size)

        # Print progress periodically
        if batch_idx % 100 == 0:
            print(
                f"Epoch: [{epoch}][{batch_idx}/{len(loader)}] "
                f"Loss: {meters['loss_total'].val:.4f} ({meters['loss_total'].avg:.4f}) "
                f"Acc: {meters['acc_species'].val:.4f} ({meters['acc_species'].avg:.4f})"
            )

    end_time = time.time()
    epoch_time = end_time - start_time

    print(
        f"Epoch {epoch} completed in {epoch_time:.0f}s. "
        f"Avg Loss: {meters['loss_total'].avg:.4f}, "
        f"Avg Species Acc: {meters['acc_species'].avg:.4f}"
    )

    return {k: v.avg for k, v in meters.items()}


def validate(model, loader, loss_fn, device, config):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: The validation DataLoader.
        loss_fn: The HierarchicalLoss function.
        device: The device to evaluate on.
        config: Configuration object.

    Returns:
        dict: Validation metrics including Macro F1.
    """
    model.eval()

    meters = {
        "loss_total": AverageMeter(),
        "loss_species": AverageMeter(),
        "loss_genus": AverageMeter(),
        "loss_family": AverageMeter(),
    }

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, species_targets, genus_targets, family_targets in loader:
            # Move data to device
            images = images.to(device, non_blocking=True)
            species_targets = species_targets.to(device, non_blocking=True)
            genus_targets = genus_targets.to(device, non_blocking=True)
            family_targets = family_targets.to(device, non_blocking=True)

            targets = (species_targets, genus_targets, family_targets)

            # Forward pass
            outputs = model(images)

            # Loss calculation
            loss, metrics = loss_fn(outputs, targets)

            # Update meters
            batch_size = images.size(0)
            meters["loss_total"].update(metrics["loss_total"], batch_size)
            meters["loss_species"].update(metrics["loss_species"], batch_size)
            meters["loss_genus"].update(metrics["loss_genus"], batch_size)
            meters["loss_family"].update(metrics["loss_family"], batch_size)

            # Store predictions for F1 score
            species_logits = outputs[0]
            preds = torch.argmax(species_logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(species_targets.cpu().numpy())

    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate Macro F1 Score
    macro_f1 = f1_score(all_targets, all_preds, average="macro")

    # Calculate Accuracy
    accuracy = (all_preds == all_targets).mean()

    # Print full precision as requested
    print(
        f"Validation Results - Loss: {meters['loss_total'].avg}, Macro F1: {macro_f1}, Accuracy: {accuracy}"
    )

    metrics = {k: v.avg for k, v in meters.items()}
    metrics["macro_f1"] = macro_f1
    metrics["accuracy"] = accuracy

    return metrics


def generate_submission(model, loader, device, config):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model: The trained PyTorch model.
        loader: The test DataLoader.
        device: The device to run inference on.
        config: Configuration object.
    """
    model.eval()
    results = []

    print("Generating predictions for submission...")
    with torch.no_grad():
        for images, image_ids in loader:
            images = images.to(device, non_blocking=True)

            # Forward pass
            # The model returns tuple (species, genus, family). We only need species (index 0).
            outputs = model(images)
            species_logits = outputs[0]

            # Get predictions
            preds = torch.argmax(species_logits, dim=1).cpu().numpy()
            ids = image_ids.numpy()

            for i, p in zip(ids, preds):
                results.append({"Id": i, "Predicted": p})

    df = pd.DataFrame(results)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
    df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")


class EarlyStopping:
    """
    Early stopping utility to stop training when validation metric stops improving.
    """

    def __init__(self, patience=3, min_delta=0.0, mode="max"):
        """
        Args:
            patience (int): How many epochs to wait after last time validation metric improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            mode (str): 'max' for metrics like Accuracy/F1, 'min' for Loss.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == "max" and score < self.best_score + self.min_delta) or (
            self.mode == "min" and score > self.best_score - self.min_delta
        ):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
