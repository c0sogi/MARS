import os
import torch
import torch.nn as nn
import numpy as np
from library.configuration import Config
from library.utilities import AverageMeter, map5


def train_fn(train_loader, model, criterion, optimizer, device, epoch):
    """
    Training loop for one epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        # For ArcFace, we must pass labels during training to calculate the margin loss
        outputs = model(images, labels)

        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def eval_fn(val_loader, model, device):
    """
    Evaluation loop with Test-Time Augmentation (TTA).
    Computes MAP@5.
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # TTA: Original images
            logits_orig = model(images, labels=None)

            # TTA: Horizontally flipped images
            images_flipped = torch.flip(images, dims=[3])
            logits_flip = model(images_flipped, labels=None)

            # Average logits
            avg_logits = (logits_orig + logits_flip) / 2.0

            # Get top 5 predictions
            # shape: (batch_size, 5)
            _, preds = torch.topk(avg_logits, 5, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate MAP@5
    score = map5(all_preds, all_targets)

    return score


def run_training(
    model, train_loader, val_loader, optimizer, scheduler, device, num_epochs
):
    """
    Main driver for training and evaluation with Early Stopping.
    """
    criterion = nn.CrossEntropyLoss()

    best_score = -1.0
    best_epoch = -1
    patience = 5
    patience_counter = 0

    save_path = os.path.join(Config.checkpoint_dir, "model_best.pth")

    print(f"Starting training for {num_epochs} epochs on device: {device}")

    for epoch in range(num_epochs):
        # 1. Train
        train_loss = train_fn(train_loader, model, criterion, optimizer, device, epoch)

        # 2. Evaluate
        val_score = eval_fn(val_loader, model, device)

        # 3. Scheduler Step
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        # 4. Print Metrics (Full Precision)
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"LR: {current_lr:.8f} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val MAP@5: {val_score}"
        )

        # 5. Checkpointing & Early Stopping
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(
                f"Early stopping triggered at epoch {epoch+1}. Best MAP@5: {best_score}"
            )
            break

    print(f"Training complete. Best MAP@5: {best_score} at epoch {best_epoch+1}")
    return best_score
