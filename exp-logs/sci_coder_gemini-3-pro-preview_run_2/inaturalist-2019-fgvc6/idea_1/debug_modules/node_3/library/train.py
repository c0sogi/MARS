import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import (
    AverageMeter,
    calculate_accuracy,
    save_checkpoint,
    load_checkpoint,
    set_seed,
)
from library.dataset import get_loaders
from library.model import MobileNetV3Baseline


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, device):
    """
    Trains the model for one epoch.
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Forward pass
        output = model(images)
        loss = criterion(output, target)

        # Measure accuracy and record loss
        acc1, acc5 = calculate_accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))
        top5.update(acc5, images.size(0))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    print(
        f"Epoch: [{epoch}] Train Loss: {losses.avg:.6f} Top1: {top1.avg:.6f} Top5: {top5.avg:.6f}"
    )
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = calculate_accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1, images.size(0))
            top5.update(acc5, images.size(0))

    # Print full precision as requested
    print(
        f"Validation Results - Loss: {losses.avg:.16f} Top1: {top1.avg:.16f} Top5: {top5.avg:.16f}"
    )
    return top1.avg


def generate_submission(model, test_loader, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    model.eval()
    predictions = []

    print("Generating submission...")

    # Ensure we have the mapping from index to category_id
    if not hasattr(test_loader.dataset, "idx_to_class"):
        raise ValueError(
            "Test dataset must have 'idx_to_class' attribute for submission."
        )

    idx_to_class = test_loader.dataset.idx_to_class

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device, non_blocking=True)

            # Forward pass
            output = model(images)

            # Get top 5 predictions
            # output is (B, NumClasses)
            _, topk_indices = torch.topk(output, k=5, dim=1)

            topk_indices = topk_indices.cpu().numpy()
            image_ids = image_ids.numpy()

            for img_id, indices in zip(image_ids, topk_indices):
                # Map model indices to original category IDs
                cat_ids = [str(idx_to_class[idx]) for idx in indices]
                # Format: "cat_id1 cat_id2 ..."
                pred_str = " ".join(cat_ids)
                predictions.append({"id": img_id, "predicted": pred_str})

    # Save to CSV
    df = pd.DataFrame(predictions)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(
    num_epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    lr=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    patience=Config.EARLY_STOPPING_PATIENCE,
):
    """
    Main function to run the training pipeline, including validation,
    early stopping, and submission generation.
    """
    # Update Config with provided arguments to ensure consistency across modules
    Config.NUM_EPOCHS = num_epochs
    Config.BATCH_SIZE = batch_size
    Config.LEARNING_RATE = lr
    Config.WEIGHT_DECAY = weight_decay
    Config.EARLY_STOPPING_PATIENCE = patience

    # Set reproducibility
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Initialize Data Loaders
    train_loader, val_loader, test_loader = get_loaders()

    # Initialize Model
    model = MobileNetV3Baseline()
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=Config.ETA_MIN
    )

    best_acc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "model_best.pth")

    print("Starting training...")
    for epoch in range(1, num_epochs + 1):
        # Train
        train_acc, train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, epoch, device
        )

        # Validate
        val_acc = validate(val_loader, model, criterion, device)

        # Update Scheduler
        scheduler.step()

        # Save Checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_acc": best_acc,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
        )

        print(f"Epoch {epoch} completed. Best Validation Accuracy: {best_acc:.16f}")

        # Early Stopping
        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {epoch} epochs. No improvement for {patience} epochs."
            )
            break

    print("Training completed.")

    # Load best model for submission
    print(f"Loading best model from {best_model_path}...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    # Generate Submission
    generate_submission(model, test_loader, device, Config.SUBMISSION_FILE_PATH)

    return best_acc
