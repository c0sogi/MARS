import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.optim.swa_utils import update_bn

from library.config import Config
from library.utils import MetricMonitor


def mixup_data(x, y, aux, alpha=0.2, device="cuda"):
    """
    Applies Mixup augmentation to inputs and targets (both class and auxiliary).
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    aux_a, aux_b = aux, aux[index]
    return mixed_x, y_a, y_b, aux_a, aux_b, lam


def mixup_criterion(
    criterion_cls, criterion_aux, pred_cls, pred_aux, y_a, y_b, aux_a, aux_b, lam
):
    """
    Computes the weighted multi-task loss with Mixup.
    L_total = L_cls + lambda_aux * L_aux
    """
    # Classification Loss (BCE)
    loss_cls = lam * criterion_cls(pred_cls, y_a.view(-1, 1)) + (
        1 - lam
    ) * criterion_cls(pred_cls, y_b.view(-1, 1))

    # Auxiliary Loss (MSE)
    loss_aux = lam * criterion_aux(pred_aux, aux_a.view(-1, 1)) + (
        1 - lam
    ) * criterion_aux(pred_aux, aux_b.view(-1, 1))

    total_loss = loss_cls + Config.AUX_LOSS_WEIGHT * loss_aux
    return total_loss, loss_cls, loss_aux


def train_one_epoch(model, train_loader, optimizer, device, epoch):
    """
    Trains the model for one epoch using Mixup and Multi-Task Loss.
    """
    model.train()
    metric_monitor = MetricMonitor(float_precision=8)

    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_aux = nn.MSELoss()

    for batch_idx, (images, labels, aux_targets) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        aux_targets = aux_targets.to(device)

        # Apply Mixup
        mixed_images, labels_a, labels_b, aux_a, aux_b, lam = mixup_data(
            images, labels, aux_targets, Config.MIXUP_ALPHA, device
        )

        optimizer.zero_grad()

        # Forward Pass
        cls_logits, aux_preds = model(mixed_images)

        # Compute Loss
        loss, loss_cls, loss_aux = mixup_criterion(
            criterion_cls,
            criterion_aux,
            cls_logits,
            aux_preds,
            labels_a,
            labels_b,
            aux_a,
            aux_b,
            lam,
        )

        # Backward Pass
        loss.backward()
        optimizer.step()

        # Update Metrics
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("ClsLoss", loss_cls.item())
        metric_monitor.update("AuxLoss", loss_aux.item())

    print(f"Epoch {epoch} Train: {metric_monitor}")
    return metric_monitor.get_avg("Loss")


def validate(model, val_loader, device):
    """
    Evaluates the model on the validation set.
    Returns metrics dict, class probabilities, and auxiliary predictions.
    """
    model.eval()
    metric_monitor = MetricMonitor(float_precision=8)

    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_aux = nn.MSELoss()

    all_preds = []
    all_targets = []
    all_aux_preds = []

    with torch.no_grad():
        for images, labels, aux_targets in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            aux_targets = aux_targets.to(device)

            # Forward Pass (No Mixup)
            cls_logits, aux_preds = model(images)

            # Compute Loss
            loss_cls = criterion_cls(cls_logits, labels.view(-1, 1))
            loss_aux = criterion_aux(aux_preds, aux_targets.view(-1, 1))
            loss = loss_cls + Config.AUX_LOSS_WEIGHT * loss_aux

            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("ClsLoss", loss_cls.item())
            metric_monitor.update("AuxLoss", loss_aux.item())

            # Collect Predictions
            preds = torch.sigmoid(cls_logits).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())
            all_aux_preds.extend(aux_preds.cpu().numpy())

    # Calculate AUC
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except Exception:
        auc = 0.5

    metric_monitor.update("AUC", auc)

    print(f"Val: {metric_monitor}")

    # Prepare return dictionary with simple float values
    metrics = {k: v["avg"] for k, v in metric_monitor.metrics.items()}

    return metrics, np.array(all_preds), np.array(all_aux_preds)


def update_swa_step(swa_model, model):
    """
    Updates the SWA model parameters with the current model's parameters.
    Should be called at the end of each epoch during the SWA phase.
    """
    swa_model.update_parameters(model)


def finalize_swa(swa_model, loader, device):
    """
    Updates Batch Normalization statistics for the SWA model.
    Should be called once at the end of training.
    """
    update_bn(loader, swa_model, device=device)
