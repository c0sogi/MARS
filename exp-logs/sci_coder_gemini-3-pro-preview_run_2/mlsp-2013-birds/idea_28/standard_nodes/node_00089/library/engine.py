import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.loss import BornAgainLoss


def mixup_data(x, y, soft_y=None, alpha=0.4, device="cuda"):
    """
    Applies Mixup augmentation to inputs and targets.
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    # Mix hard targets
    y_a, y_b = y, y[index]
    mixed_y = lam * y_a + (1 - lam) * y_b

    # Mix soft targets if they exist
    mixed_soft_y = None
    if soft_y is not None and soft_y.numel() > 0:
        soft_y_a, soft_y_b = soft_y, soft_y[index]
        mixed_soft_y = lam * soft_y_a + (1 - lam) * soft_y_b

    return mixed_x, mixed_y, mixed_soft_y


def train_one_epoch(model, dataloader, optimizer, device, epoch, loss_fn=None):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Default loss function if not provided
    if loss_fn is None:
        loss_fn = BornAgainLoss()

    for batch in dataloader:
        images = batch["image"].to(device)
        targets = batch["targets"].to(device)
        soft_targets = batch["soft_targets"].to(device)

        batch_size = images.size(0)

        # Apply Mixup
        if Config.MIXUP_ALPHA > 0:
            images, targets, soft_targets = mixup_data(
                images, targets, soft_targets, alpha=Config.MIXUP_ALPHA, device=device
            )

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)

        # Calculate loss
        # soft_targets might be None or empty tensor if not provided in dataset
        if soft_targets is not None and soft_targets.numel() > 0:
            loss = loss_fn(logits, targets, soft_targets)
        else:
            loss = loss_fn(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, device, loss_fn=None):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    if loss_fn is None:
        loss_fn = BornAgainLoss()

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = batch["targets"].to(device)

            batch_size = images.size(0)

            logits = model(images)

            # For validation, we typically calculate loss against hard targets only
            loss = loss_fn(logits, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(logits)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    avg_loss = running_loss / dataset_size

    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # Calculate ROC AUC
    # Handle potential edge cases where a class might not be present in the validation set
    try:
        roc_auc = roc_auc_score(all_targets, all_preds, average="macro")
    except ValueError:
        # Fallback if a class is missing in targets
        roc_auc = 0.5

    return avg_loss, roc_auc


def predict(model, dataloader, device):
    """
    Generates predictions for a given dataloader.
    Returns recording IDs and predicted probabilities.
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            rec_ids = batch["rec_id"]

            logits = model(images)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_ids.extend(rec_ids.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_ids = np.array(all_ids)

    return all_ids, all_preds
