import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import AverageMeter, calculate_trust_score


def train_expert_one_epoch(
    train_loader, model, criterion_cls, criterion_aux, optimizer, device, epoch
):
    """
    Trains an expert model for one epoch using Multi-Task Loss and Mixup.
    """
    model.train()

    losses = AverageMeter("Loss", ":.4f")
    cls_losses = AverageMeter("Cls Loss", ":.4f")
    aux_losses = AverageMeter("Aux Loss", ":.4f")

    for i, batch in enumerate(train_loader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device).float().view(-1, 1)
        log_sizes = batch["log_size"].to(device).float().view(-1, 1)

        # Mixup
        if Config.USE_MIXUP and Config.MIXUP_ALPHA > 0:
            lam = np.random.beta(Config.MIXUP_ALPHA, Config.MIXUP_ALPHA)
            index = torch.randperm(images.size(0)).to(device)

            mixed_images = lam * images + (1 - lam) * images[index, :]

            # Mix targets for both tasks
            mixed_labels = lam * labels + (1 - lam) * labels[index, :]
            mixed_log_sizes = lam * log_sizes + (1 - lam) * log_sizes[index, :]

            # Forward pass
            logits, aux_out = model(mixed_images)

            # Compute losses with mixed targets
            loss_cls = criterion_cls(logits, mixed_labels)
            loss_aux = criterion_aux(aux_out, mixed_log_sizes)

        else:
            # Standard training
            logits, aux_out = model(images)
            loss_cls = criterion_cls(logits, labels)
            loss_aux = criterion_aux(aux_out, log_sizes)

        # Weighted Multi-Task Loss
        loss = loss_cls + Config.AUX_LOSS_WEIGHT * loss_aux

        # Backward and Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))
        cls_losses.update(loss_cls.item(), images.size(0))
        aux_losses.update(loss_aux.item(), images.size(0))

    return losses.avg, cls_losses.avg, aux_losses.avg


def validate_expert(val_loader, model, criterion_cls, criterion_aux, device):
    """
    Validates an expert model. Returns metrics and predictions for OOF generation.
    """
    model.eval()

    losses = AverageMeter("Loss", ":.4f")
    cls_losses = AverageMeter("Cls Loss", ":.4f")
    aux_losses = AverageMeter("Aux Loss", ":.4f")

    all_preds = []
    all_aux_preds = []
    all_targets = []
    all_aux_targets = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).float().view(-1, 1)
            log_sizes = batch["log_size"].to(device).float().view(-1, 1)

            logits, aux_out = model(images)

            loss_cls = criterion_cls(logits, labels)
            loss_aux = criterion_aux(aux_out, log_sizes)
            loss = loss_cls + Config.AUX_LOSS_WEIGHT * loss_aux

            losses.update(loss.item(), images.size(0))
            cls_losses.update(loss_cls.item(), images.size(0))
            aux_losses.update(loss_aux.item(), images.size(0))

            # Store predictions (sigmoid for class, raw for aux)
            preds = torch.sigmoid(logits)

            all_preds.append(preds.cpu().numpy())
            all_aux_preds.append(aux_out.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            all_aux_targets.append(log_sizes.cpu().numpy())

    # Concatenate results
    all_preds = np.concatenate(all_preds)
    all_aux_preds = np.concatenate(all_aux_preds)
    all_targets = np.concatenate(all_targets)
    all_aux_targets = np.concatenate(all_aux_targets)

    # Calculate AUC
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return {
        "loss": losses.avg,
        "cls_loss": cls_losses.avg,
        "aux_loss": aux_losses.avg,
        "auc": auc,
        "preds": all_preds,
        "aux_preds": all_aux_preds,
        "targets": all_targets,
        "aux_targets": all_aux_targets,
    }


def update_swa_bn(loader, model, device):
    """
    Updates BatchNorm statistics for the SWA model.
    """
    model.train()
    # Reset BN running stats
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            module.momentum = None  # Use simple average

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            model(images)


def train_router_epoch(
    trust_scores, expert_preds, labels, router, criterion, optimizer, device
):
    """
    Trains the gating network (router) for one epoch.

    Args:
        trust_scores: Tensor (N, Num_Experts) - Absolute aux errors
        expert_preds: Tensor (N, Num_Experts) - Predicted probabilities
        labels: Tensor (N, 1) - Ground truth
    """
    router.train()

    # Move full dataset to device (usually small enough for router training)
    trust_scores = trust_scores.to(device)
    expert_preds = expert_preds.to(device)
    labels = labels.to(device)

    # Shuffle indices
    indices = torch.randperm(trust_scores.size(0)).to(device)

    # Mini-batch training for Router
    batch_size = Config.BATCH_SIZE
    num_samples = trust_scores.size(0)
    num_batches = (num_samples + batch_size - 1) // batch_size

    losses = AverageMeter("Router Loss", ":.4f")

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        batch_idx = indices[start_idx:end_idx]

        batch_trust = trust_scores[batch_idx]
        batch_preds = expert_preds[batch_idx]
        batch_labels = labels[batch_idx]

        # Forward Pass
        # Get weights from router based on trust scores
        weights = router(batch_trust)  # (B, K)

        # Weighted ensemble prediction
        # sum(weights * preds) -> (B, 1)
        # Note: expert_preds are already probabilities (sigmoid applied)
        weighted_pred = torch.sum(weights * batch_preds, dim=1, keepdim=True)

        # Loss (BCE on the ensemble probability)
        # We clamp to avoid log(0) issues, though BCE usually handles logits.
        # Since we are combining probabilities, we use BCELoss, not BCEWithLogitsLoss
        loss = criterion(weighted_pred, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), len(batch_idx))

    return losses.avg


def validate_router(trust_scores, expert_preds, labels, router, criterion, device):
    """
    Validates the router.
    """
    router.eval()

    trust_scores = trust_scores.to(device)
    expert_preds = expert_preds.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        weights = router(trust_scores)
        weighted_pred = torch.sum(weights * expert_preds, dim=1, keepdim=True)
        loss = criterion(weighted_pred, labels)

    preds_np = weighted_pred.cpu().numpy()
    labels_np = labels.cpu().numpy()

    try:
        auc = roc_auc_score(labels_np, preds_np)
    except ValueError:
        auc = 0.5

    return loss.item(), auc
