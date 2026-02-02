import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import AverageMeter, calculate_f1_score


def train_one_epoch(model, loader, optimizer, device, mixup_fn, loss_fn):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: Training DataLoader.
        optimizer: The optimizer.
        device: Computation device (cpu or cuda).
        mixup_fn: MixupCutmix callable or None.
        loss_fn: Loss function (AsymmetricLoss).

    Returns:
        float: Average training loss.
    """
    model.train()
    losses = AverageMeter()

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        # Apply Mixup/Cutmix augmentation
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)

        # Compute loss
        loss = loss_fn(logits, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, loss_fn, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: Validation DataLoader.
        loss_fn: Loss function.
        device: Computation device.

    Returns:
        tuple: (Average Validation Loss, Validation F1 Score)
    """
    model.eval()
    losses = AverageMeter()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = loss_fn(logits, targets)

            losses.update(loss.item(), images.size(0))

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate results from all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate F1 Score
    f1 = calculate_f1_score(all_targets, all_preds)

    return losses.avg, f1


def inference_tta(model, loader, device, output_path):
    """
    Generates predictions using Test Time Augmentation (TTA).

    Args:
        model: The trained PyTorch model.
        loader: Test DataLoader (returns image, image_id).
        device: Computation device.
        output_path: Path to save the submission CSV.

    Returns:
        pd.DataFrame: The submission dataframe.
    """
    model.eval()
    results = []

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with torch.no_grad():
        for images, image_ids in loader:
            images = images.to(device)

            # 1. Forward pass on original images
            logits_orig = model(images)

            # 2. Forward pass on horizontally flipped images (TTA)
            # Flip along width dimension (dim 3 for NCHW)
            images_flipped = torch.flip(images, dims=[3])
            logits_flipped = model(images_flipped)

            # 3. Average logits
            avg_logits = (logits_orig + logits_flipped) / 2.0

            # 4. Apply Sigmoid and threshold
            probs = torch.sigmoid(avg_logits)
            probs = probs.cpu().numpy()

            # 5. Convert to labels
            for i, img_id in enumerate(image_ids):
                pred_row = probs[i]
                labels = []
                for idx, score in enumerate(pred_row):
                    if score > 0.5:
                        labels.append(Config.CLASSES[idx])

                # Join labels with space
                label_str = " ".join(labels)

                results.append({"image": img_id, "labels": label_str})

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    return df


class EarlyStopping:
    """
    Early stopping to stop training when the validation metric has not improved.
    """

    def __init__(self, patience=3, delta=0.0, path="checkpoint.pth"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score, model):
        # Assuming we want to maximize the score (F1)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    mixup_fn,
    loss_fn,
    patience,
    save_path,
):
    """
    Main training loop orchestrator.

    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device.
        epochs: Total epochs.
        mixup_fn: Mixup function.
        loss_fn: Loss function.
        patience: Early stopping patience.
        save_path: Path to save best model.

    Returns:
        model: The model with the best weights loaded.
    """
    early_stopping = EarlyStopping(patience=patience, path=save_path)

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, mixup_fn, loss_fn
        )

        # Validate
        val_loss, val_f1 = validate(model, val_loader, loss_fn, device)

        # Step Scheduler
        if scheduler is not None:
            scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val F1: {val_f1}"
        )

        # Check Early Stopping
        early_stopping(val_f1, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Load best model weights
    print(f"Loading best model from {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device))

    return model
