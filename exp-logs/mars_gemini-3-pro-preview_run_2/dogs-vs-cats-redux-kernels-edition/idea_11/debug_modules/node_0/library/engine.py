import numpy as np
import torch
import torch.nn as nn
from library.utils import MetricMonitor, calculate_log_loss
from library.config import Config

# =============================================================================
# Augmentation Helpers (Mixup & CutMix)
# =============================================================================


def rand_bbox(size, lam):
    """Generates a random bounding box for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniformly sample the center of the box
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """Performs Mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0, device="cuda"):
    """Performs CutMix augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    y_a, y_b = y, y[index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to be the exact pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    return x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Calculates loss for mixed targets."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =============================================================================
# Core Engine Functions
# =============================================================================


def train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch):
    """
    Trains the model for one epoch using Mixup and CutMix.
    """
    model.train()
    metric_monitor = MetricMonitor()
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Randomly apply Mixup or CutMix (50% chance each)
        # This acts as strong regularization and improves calibration
        choice = np.random.rand()
        if choice < 0.5:
            images, y_a, y_b, lam = mixup_data(images, labels, alpha=1.0, device=device)
            outputs = model(images)
            outputs = outputs.squeeze(1)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            images, y_a, y_b, lam = cutmix_data(
                images, labels, alpha=1.0, device=device
            )
            outputs = model(images)
            outputs = outputs.squeeze(1)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

        metric_monitor.update("Loss", loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Step the scheduler at the end of the epoch
    if scheduler is not None:
        scheduler.step()

    print(f"Epoch {epoch} Train: {metric_monitor}")
    return metric_monitor.get_avg("Loss")


def valid_one_epoch(model, val_loader, device, epoch):
    """
    Validates the model on the validation set.
    """
    model.eval()
    metric_monitor = MetricMonitor()
    criterion = nn.BCEWithLogitsLoss()

    preds = []
    targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            outputs = outputs.squeeze(1)

            loss = criterion(outputs, labels)
            metric_monitor.update("Loss", loss.item())

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            preds.extend(probs.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    # Calculate Log Loss metric
    log_loss_val = calculate_log_loss(targets, preds)
    metric_monitor.update("LogLoss", log_loss_val)

    print(f"Epoch {epoch} Valid: {metric_monitor}")
    return metric_monitor.get_avg("Loss"), log_loss_val


def inference_fn(model, test_loader, device):
    """
    Generates predictions with Test-Time Augmentation (TTA).
    Averages predictions from original and horizontally flipped images.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in test_loader:
            # Handle cases where loader returns (img, id) or just img
            if isinstance(batch, (tuple, list)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device)

            # 1. Forward pass on original images
            out_orig = model(images)
            prob_orig = torch.sigmoid(out_orig).squeeze(1)

            # 2. Forward pass on horizontally flipped images (TTA)
            images_flip = torch.flip(images, dims=[3])
            out_flip = model(images_flip)
            prob_flip = torch.sigmoid(out_flip).squeeze(1)

            # 3. Average probabilities
            avg_prob = (prob_orig + prob_flip) / 2.0

            preds.extend(avg_prob.cpu().numpy())

    return np.array(preds)


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    patience,
    save_path,
):
    """
    Orchestrates the training process with Early Stopping.
    """
    best_log_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        _ = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        _, val_log_loss = valid_one_epoch(model, val_loader, device, epoch)

        # Check for improvement
        if val_log_loss < best_log_loss:
            best_log_loss = val_log_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model at epoch {epoch} with LogLoss: {best_log_loss}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    return best_log_loss
