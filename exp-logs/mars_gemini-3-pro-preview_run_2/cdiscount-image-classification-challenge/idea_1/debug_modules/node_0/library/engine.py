import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import DEVICE, SUBMISSION_FILE, CHECKPOINT_DIR
from library.utils import AverageMeter, accuracy, save_checkpoint


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): How many epochs to wait before stopping when loss is
                            not improving.
            min_delta (float): Minimum difference between new loss and old loss for
                               new loss to be considered as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_one_epoch(
    train_loader, model, criterion, optimizer, scaler, epoch, device=DEVICE
):
    """
    Trains the model for one epoch using Mixed Precision.
    """
    model.train()

    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")

    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed Precision Context
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, targets)

        # Scale loss and backprop
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        acc1 = accuracy(output, targets, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, device=DEVICE):
    """
    Evaluates the model on the validation set using Late Fusion (averaging probabilities).
    """
    model.eval()

    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")

    with torch.no_grad():
        for i, (flattened_images, targets, product_ids, num_imgs) in enumerate(
            val_loader
        ):
            flattened_images = flattened_images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Inference on all images in the batch
            with torch.cuda.amp.autocast():
                output = model(flattened_images)

                # Calculate image-level loss for monitoring
                # Expand targets to match flattened images
                img_targets = []
                for t, n in zip(targets, num_imgs):
                    img_targets.extend([t.item()] * n.item())
                img_targets = torch.tensor(img_targets, device=device, dtype=torch.long)

                loss = criterion(output, img_targets)

            losses.update(loss.item(), flattened_images.size(0))

            # Late Fusion: Average probabilities per product
            probs = torch.softmax(output, dim=1)

            # Split flattened probabilities back into per-product groups
            split_probs = torch.split(probs, num_imgs.tolist())

            batch_preds = []
            for prod_probs in split_probs:
                # Average probabilities across images of the product
                avg_prob = torch.mean(prod_probs, dim=0)
                pred_cat = torch.argmax(avg_prob).item()
                batch_preds.append(pred_cat)

            batch_preds = torch.tensor(batch_preds, device=device)

            # Calculate product-level accuracy
            acc = (batch_preds == targets).float().mean() * 100.0
            top1.update(acc.item(), targets.size(0))

    print(f"Validation Results - Top1: {top1.avg} Loss: {losses.avg}")
    return top1.avg, losses.avg


def make_predictions(test_loader, model, device=DEVICE, output_file=SUBMISSION_FILE):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    model.eval()
    results = []

    # Retrieve category mapping
    if hasattr(test_loader.dataset, "idx_to_cat"):
        idx_to_cat = test_loader.dataset.idx_to_cat
    else:
        raise AttributeError("Dataset does not have idx_to_cat mapping.")

    print("Generating predictions...")
    with torch.no_grad():
        for i, (flattened_images, _, product_ids, num_imgs) in enumerate(test_loader):
            flattened_images = flattened_images.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                output = model(flattened_images)
                probs = torch.softmax(output, dim=1)

            split_probs = torch.split(probs, num_imgs.tolist())

            for j, prod_probs in enumerate(split_probs):
                # Late Fusion
                avg_prob = torch.mean(prod_probs, dim=0)
                pred_idx = torch.argmax(avg_prob).item()
                pred_cat_id = idx_to_cat[pred_idx]

                p_id = product_ids[j].item()
                results.append((p_id, pred_cat_id))

    # Save to CSV
    df = pd.DataFrame(results, columns=["_id", "category_id"])
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=10,
    device=DEVICE,
    patience=3,
):
    """
    Orchestrates the training process, including training loops, validation, checkpoints, and early stopping.
    """
    scaler = torch.cuda.amp.GradScaler()
    early_stopping = EarlyStopping(patience=patience)

    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Train
        train_acc, train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, scaler, epoch, device
        )
        print(f"Train Loss: {train_loss} Train Acc: {train_acc}")

        # Validate
        val_acc, val_loss = validate(val_loader, model, criterion, device)

        # Scheduler Step
        if scheduler:
            # Handle different scheduler types
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Save Checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_acc1": best_acc,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
        )

        # Early Stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    return model
