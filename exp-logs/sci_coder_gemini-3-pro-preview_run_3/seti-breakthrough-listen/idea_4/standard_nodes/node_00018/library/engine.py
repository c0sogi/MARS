import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from library.config import Config
from library.utils import AverageMeter, get_score


class EarlyStopping:
    """
    Early stopping to stop the training when the metric does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, mode="max", delta=0.0, save_path=None):
        """
        Args:
            patience (int): How long to wait after last time validation metric improved.
            mode (str): 'min' for loss, 'max' for metric (AUC).
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            save_path (str): Path to save the best model.
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score = np.inf if mode == "min" else -np.inf

    def __call__(self, epoch_score, model):
        score = epoch_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model)
        else:
            if self.mode == "min":
                if score < self.best_score - self.delta:
                    self.best_score = score
                    self.save_checkpoint(epoch_score, model)
                    self.counter = 0
                else:
                    self.counter += 1
            else:  # mode == 'max'
                if score > self.best_score + self.delta:
                    self.best_score = score
                    self.save_checkpoint(epoch_score, model)
                    self.counter = 0
                else:
                    self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

    def save_checkpoint(self, epoch_score, model):
        """Saves model when validation score improves."""
        if self.save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(model.state_dict(), self.save_path)
        self.val_score = epoch_score


def mixup_data(x, target, alpha=1.0, device="cuda"):
    """
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    target_a, target_b = target, target[index]
    return mixed_x, target_a, target_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    for batch in data_loader:
        x = batch["input"].to(device)
        target = batch["target"].to(device).unsqueeze(1)

        if Config.MIXUP_ALPHA > 0:
            x, target_a, target_b, lam = mixup_data(
                x, target, Config.MIXUP_ALPHA, device
            )
            logits = model(x)
            loss = mixup_criterion(criterion, logits, target_a, target_b, lam)
        else:
            logits = model(x)
            loss = criterion(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), x.size(0))

    print(f"Epoch {epoch} Train Loss: {losses.avg}")
    return losses.avg


def validate(model, data_loader, device):
    """
    Validates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    preds = []
    valid_labels = []

    with torch.no_grad():
        for batch in data_loader:
            x = batch["input"].to(device)
            target = batch["target"].to(device).unsqueeze(1)

            logits = model(x)
            loss = criterion(logits, target)

            losses.update(loss.item(), x.size(0))

            probs = torch.sigmoid(logits)
            preds.extend(probs.cpu().numpy().flatten())
            valid_labels.extend(target.cpu().numpy().flatten())

    preds = np.array(preds)
    valid_labels = np.array(valid_labels)
    auc_score = get_score(valid_labels, preds)

    print(f"Validation Loss: {losses.avg}")
    print(f"Validation AUC: {auc_score}")

    return losses.avg, auc_score


def generate_submission(model, data_loader, device, output_path):
    """
    Generates predictions for the test set using TTA and saves to CSV.
    """
    model.eval()
    preds = []
    ids = []

    with torch.no_grad():
        for batch in data_loader:
            x = batch["input"].to(device)
            batch_ids = batch["id"]

            # 1. Original Pass
            logits = model(x)
            probs = torch.sigmoid(logits)

            # 2. TTA: Horizontal Flip (Time Reversal)
            # Input shape: (B, 6, H, W). Flip width (dim 3).
            x_hflip = torch.flip(x, dims=[3])
            logits_hflip = model(x_hflip)
            probs_hflip = torch.sigmoid(logits_hflip)

            # 3. TTA: Vertical Flip (Frequency Inversion)
            # Cite solution_lesson_node_00014
            # Flip height (dim 2)
            x_vflip = torch.flip(x, dims=[2])
            logits_vflip = model(x_vflip)
            probs_vflip = torch.sigmoid(logits_vflip)

            # Average predictions
            avg_probs = (probs + probs_hflip + probs_vflip) / 3.0

            preds.extend(avg_probs.cpu().numpy().flatten())
            ids.extend(batch_ids)

    df = pd.DataFrame({"id": ids, "target": preds})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
