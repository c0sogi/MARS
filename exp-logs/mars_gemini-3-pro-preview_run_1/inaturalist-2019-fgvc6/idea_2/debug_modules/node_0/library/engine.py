import os
import torch
import pandas as pd
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from library.utils import AverageMeter, accuracy, seed_everything
from library.dataset import INatDataset, get_transforms


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0, path="./working/idea_2/best_model.pth"):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        torch.save(model.state_dict(), self.path)


def train_one_epoch(
    model, loader, criterion, optimizer, device, scaler, epoch, max_steps=None
):
    """
    Trains the model for one epoch.

    Args:
        max_steps (int, optional): Limit the number of steps for debugging.
    """
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()

    for i, (images, target) in enumerate(loader):
        if max_steps and i >= max_steps:
            break

        images = images.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        with autocast():
            output = model(images)
            loss = criterion(output, target)

        scaler.scale(loss).backward()

        # Unscale before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        scaler.step(optimizer)
        scaler.update()

        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

    print(f"Epoch: {epoch} | Train Loss: {losses.avg} | Train Acc: {top1.avg}")
    return losses.avg, top1.avg


def validate(model, loader, criterion, device, max_steps=None):
    """
    Validates the model.

    Args:
        max_steps (int, optional): Limit the number of steps for debugging.
    """
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            if max_steps and i >= max_steps:
                break

            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)

            acc1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))

    print(f"Val Loss: {losses.avg} | Val Acc: {top1.avg}")
    return losses.avg, top1.avg


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    checkpoint_path,
    patience=5,
    debug_limit=None,
):
    """
    Main training loop with early stopping.

    Args:
        debug_limit (int, optional): If set, limits the number of batches per epoch.
    """
    seed_everything(42)
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=patience, path=checkpoint_path)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            epoch,
            max_steps=debug_limit,
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion, device, max_steps=debug_limit
        )

        if scheduler:
            scheduler.step()

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Load the best model weights
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded best model from {checkpoint_path}")

    return model


def generate_predictions(
    model,
    device,
    test_csv_path="./metadata/test.csv",
    output_csv_path="./submission/submission.csv",
    batch_size=64,
    num_workers=4,
    debug_limit=None,
):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        debug_limit (int, optional): If set, only processes the first N samples.
    """
    seed_everything(42)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Load metadata
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test metadata not found at {test_csv_path}")

    df = pd.read_csv(test_csv_path)
    if debug_limit:
        df = df.head(debug_limit)
        print(f"Debugging: Limiting test set to {debug_limit} samples.")

    # Create dataset and loader
    dataset = INatDataset(df, transform=get_transforms("test"), is_test=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model.eval()
    results = []

    print("Generating predictions...")
    with torch.no_grad():
        for i, (images, image_ids) in enumerate(loader):
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Get top 5 predictions (indices)
            # outputs shape: (B, NumClasses)
            _, preds = outputs.topk(5, 1, True, True)

            preds = preds.cpu().numpy()
            image_ids = image_ids.numpy()

            # Format predictions
            for img_id, pred_row in zip(image_ids, preds):
                # Join top 5 class indices with spaces
                pred_str = " ".join(map(str, pred_row))
                results.append({"id": img_id, "predicted": pred_str})

    # Create submission DataFrame
    sub_df = pd.DataFrame(results)
    sub_df.to_csv(output_csv_path, index=False)
    print(f"Submission saved to {output_csv_path}")
