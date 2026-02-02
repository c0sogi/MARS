import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import calculate_roc_auc


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (inputs, targets, _) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        if Config.USE_MIXUP and Config.MIXUP_ALPHA > 0:
            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, targets, Config.MIXUP_ALPHA, use_cuda=device.type == "cuda"
            )
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        dataset_size += inputs.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            dataset_size += inputs.size(0)

            # Apply sigmoid to get probabilities for AUC calculation
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.cpu())
            all_preds.append(probs.cpu())

    epoch_loss = running_loss / dataset_size

    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()

    auc_score = calculate_roc_auc(all_targets, all_preds)

    print(f"Validation Loss: {epoch_loss}")
    print(f"Validation ROC AUC: {auc_score}")

    return epoch_loss, auc_score


def predict_with_tta(model, loader, device):
    """
    Generates predictions using Test-Time Augmentation (TTA).
    Averages predictions from: Original, Horizontal Flip, Vertical Flip.

    Returns:
        dict: mapping from image_id (str) to probability (float)
    """
    model.eval()
    predictions = {}

    with torch.no_grad():
        for inputs, ids in loader:
            inputs = inputs.to(device)

            # 1. Original
            out_orig = model(inputs)
            prob_orig = torch.sigmoid(out_orig)

            # 2. Horizontal Flip
            inputs_h = torch.flip(inputs, [3])
            out_h = model(inputs_h)
            prob_h = torch.sigmoid(out_h)

            # 3. Vertical Flip
            inputs_v = torch.flip(inputs, [2])
            out_v = model(inputs_v)
            prob_v = torch.sigmoid(out_v)

            # Average probabilities
            avg_prob = (prob_orig + prob_h + prob_v) / 3.0
            avg_prob = avg_prob.cpu().numpy().flatten()

            for i, img_id in enumerate(ids):
                predictions[img_id] = float(avg_prob[i])

    return predictions
