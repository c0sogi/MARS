import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import AverageMeter, mixup_data


def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Trains the model for one epoch using Mixup augmentation.
    """
    model.train()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Apply Mixup
        # mixup_data returns mixed inputs, pairs of targets, and lambda
        mixed_images, labels_a, labels_b, lam = mixup_data(
            images, labels, Config.MIXUP_ALPHA, device
        )

        # Forward pass
        outputs = model(mixed_images)
        # Squeeze output from (B, 1) to (B,) to match labels
        outputs = outputs.squeeze(1)

        # Compute Mixup loss
        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(
            outputs, labels_b
        )

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))

    print(f"Epoch {epoch}: Train Loss = {losses.avg}")
    return losses.avg


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            outputs = outputs.squeeze(1)

            # Compute loss
            loss = criterion(outputs, labels)

            # Update metrics
            losses.update(loss.item(), images.size(0))

    # Print full precision as requested
    print(f"Validation Loss = {losses.avg}")
    return losses.avg


def fit(model, train_loader, val_loader, optimizer, scheduler, device, checkpoint_path):
    """
    Orchestrates the training loop for the specified number of epochs.
    Saves the best model checkpoint based on validation loss.
    """
    best_loss = float("inf")

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_loss = validate(model, val_loader, device)

        # Step Scheduler
        if scheduler is not None:
            scheduler.step()

        # Checkpoint: Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

    return best_loss


def predict(model, loader, device):
    """
    Generates predictions for the test set using Test-Time Augmentation (TTA).
    Returns a list of dictionaries with 'id' and 'label' (probability).
    """
    model.eval()
    results = []

    with torch.no_grad():
        for batch_idx, (images, ids) in enumerate(loader):
            images = images.to(device)

            # Original prediction
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze(1)

            # Test-Time Augmentation (Horizontal Flip)
            if Config.TTA_FLIP:
                # Flip images horizontally (dim 3 is width)
                images_flipped = torch.flip(images, dims=[3])
                outputs_flipped = model(images_flipped)
                probs_flipped = torch.sigmoid(outputs_flipped).squeeze(1)

                # Average predictions
                probs = (probs + probs_flipped) / 2.0

            # Convert to numpy
            probs = probs.cpu().numpy()
            ids = ids.numpy()

            # Store results
            for i in range(len(ids)):
                results.append({"id": ids[i], "label": probs[i]})

    return results
