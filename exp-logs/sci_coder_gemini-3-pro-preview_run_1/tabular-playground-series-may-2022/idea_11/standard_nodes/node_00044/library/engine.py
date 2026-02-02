import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from library.utils import MetricMonitor


class LabelSmoothingBCE(nn.Module):
    """
    Binary Cross Entropy with Label Smoothing.
    """

    def __init__(self, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # targets are 0 or 1.
        # smoothed: target * (1 - alpha) + 0.5 * alpha
        # if target=1: 1 - alpha + 0.5*alpha = 1 - 0.5*alpha
        # if target=0: 0 + 0.5*alpha = 0.5*alpha
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets_smooth)


def train_fn(model, data_loader, optimizer, scheduler, device, config):
    """
    Executes one training epoch.
    """
    model.train()
    monitor = MetricMonitor()

    # Loss functions
    criterion_main = LabelSmoothingBCE(smoothing=config.LABEL_SMOOTHING)
    criterion_recon_seq = nn.CrossEntropyLoss()
    criterion_recon_num = nn.MSELoss()

    for batch in data_loader:
        # Move data to device
        x_num = batch["x_num"].to(device)
        x_seq = batch["x_seq"].to(device)
        target = batch["target"].to(device)
        target_seq = batch["target_seq"].to(device)
        target_num = batch["target_num"].to(device)
        mask_seq = batch["mask_seq"].to(device)
        mask_num = batch["mask_num"].to(device)

        optimizer.zero_grad()

        # Forward pass
        logits, recon_logits_seq, recon_preds_num = model(x_num, x_seq)

        # 1. Main Task Loss (Binary Classification)
        loss_bce = criterion_main(logits.squeeze(), target)

        # 2. Auxiliary Task Loss (Sequence Reconstruction)
        if mask_seq.sum() > 0:
            active_logits = recon_logits_seq[mask_seq]
            active_targets = target_seq[mask_seq]
            loss_recon_seq = criterion_recon_seq(active_logits, active_targets)
        else:
            loss_recon_seq = torch.tensor(0.0, device=device)

        # 3. Auxiliary Task Loss (Numerical Reconstruction)
        if mask_num.sum() > 0:
            active_preds = recon_preds_num[mask_num]
            active_targets = target_num[mask_num]
            loss_recon_num = criterion_recon_num(active_preds, active_targets)
        else:
            loss_recon_num = torch.tensor(0.0, device=device)

        # Composite Loss
        loss = loss_bce + config.RECON_LAMBDA * (loss_recon_seq + loss_recon_num)

        # Backward pass
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        monitor.update("loss", loss.item())

    return monitor.get("loss")


def eval_fn(model, data_loader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    monitor = MetricMonitor()
    criterion = nn.BCEWithLogitsLoss()

    preds = []
    targets = []

    with torch.no_grad():
        for batch in data_loader:
            x_num = batch["x_num"].to(device)
            x_seq = batch["x_seq"].to(device)
            target = batch["target"].to(device)

            # Forward pass
            logits, _, _ = model(x_num, x_seq)

            loss = criterion(logits.squeeze(), target)
            monitor.update("loss", loss.item())

            # Collect predictions for AUC
            preds.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
            targets.extend(target.cpu().numpy())

    avg_loss = monitor.get("loss")
    auc = roc_auc_score(targets, preds)

    return avg_loss, auc


def predict_fn(model, data_loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in data_loader:
            x_num = batch["x_num"].to(device)
            x_seq = batch["x_seq"].to(device)

            logits, _, _ = model(x_num, x_seq)
            preds.extend(torch.sigmoid(logits).squeeze().cpu().numpy())

    return np.array(preds)
