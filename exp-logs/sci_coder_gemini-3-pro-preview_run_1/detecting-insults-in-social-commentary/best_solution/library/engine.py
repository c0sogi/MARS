import torch
import torch.nn as nn
import numpy as np
import time
from library.config import Config
from library.utils import get_logger, get_auc_score
from library.awp import AWP


def train_fn(train_loader, model, optimizer, epoch, scheduler, device, awp=None):
    """
    Training loop for one epoch.
    Handles forward pass, loss calculation, AWP, backprop, and optimization.

    Args:
        train_loader: DataLoader for training data.
        model: The HybridDebertaModel.
        optimizer: PyTorch optimizer.
        epoch: Current epoch number.
        scheduler: Learning rate scheduler.
        device: Torch device.
        awp: Instance of AWP class (optional).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    losses = []

    # BCEWithLogitsLoss works for both binary targets (0/1) and soft targets (probs)
    criterion = nn.BCEWithLogitsLoss()

    start_time = time.time()

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        svd_features = batch["svd_features"].to(device)
        targets = batch["target"].to(device)

        batch_size = input_ids.size(0)

        # Mixed Precision Training
        with torch.cuda.amp.autocast(enabled=True):
            y_preds = model(input_ids, attention_mask, svd_features)
            loss = criterion(y_preds.view(-1), targets)

        # Normalize loss for gradient accumulation
        if Config.gradient_accumulation_steps > 1:
            loss = loss / Config.gradient_accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()

        # AWP Attack (if enabled and initialized)
        if awp is not None:
            # The awp.step() method checks the start_epoch internally
            if awp.step(epoch):
                # Clear gradients for the adversarial step (optional, but standard AWP accumulates or re-computes)
                # Here we follow the standard pattern: forward -> backward -> restore
                with torch.cuda.amp.autocast(enabled=True):
                    y_preds_adv = model(input_ids, attention_mask, svd_features)
                    loss_adv = criterion(y_preds_adv.view(-1), targets)

                if Config.gradient_accumulation_steps > 1:
                    loss_adv = loss_adv / Config.gradient_accumulation_steps

                scaler.scale(loss_adv).backward()
                awp.restore()

        # Optimization Step
        if (step + 1) % Config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        losses.append(loss.item() * Config.gradient_accumulation_steps)

    avg_loss = np.mean(losses)
    return avg_loss


def eval_fn(val_loader, model, device):
    """
    Evaluation loop.
    Computes loss and AUC on the validation set.

    Args:
        val_loader: DataLoader for validation data.
        model: The HybridDebertaModel.
        device: Torch device.

    Returns:
        tuple: (average_loss, predictions, true_labels)
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    losses = []
    preds = []
    labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            svd_features = batch["svd_features"].to(device)
            targets = batch["target"].to(device)

            with torch.cuda.amp.autocast(enabled=True):
                y_preds = model(input_ids, attention_mask, svd_features)
                loss = criterion(y_preds.view(-1), targets)

            losses.append(loss.item())

            # Apply sigmoid to convert logits to probabilities
            preds.append(torch.sigmoid(y_preds).detach().cpu().numpy())
            labels.append(targets.detach().cpu().numpy())

    avg_loss = np.mean(losses)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    return avg_loss, preds, labels


def inference_fn(test_loader, model, device):
    """
    Inference loop for generating predictions on test data.

    Args:
        test_loader: DataLoader for test data.
        model: The HybridDebertaModel.
        device: Torch device.

    Returns:
        np.array: Predicted probabilities.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            svd_features = batch["svd_features"].to(device)

            with torch.cuda.amp.autocast(enabled=True):
                y_preds = model(input_ids, attention_mask, svd_features)

            # Apply sigmoid to convert logits to probabilities
            preds.append(torch.sigmoid(y_preds).detach().cpu().numpy())

    preds = np.concatenate(preds).flatten()
    return preds


def get_optimizer_params(model):
    """
    Configures differential learning rates for the backbone and the head.

    Args:
        model: The HybridDebertaModel.

    Returns:
        list: List of parameter groups for the optimizer.
    """
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        # Backbone parameters
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" in n and not any(nd in n for nd in no_decay)
            ],
            "lr": Config.lr_backbone,
            "weight_decay": Config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" in n and any(nd in n for nd in no_decay)
            ],
            "lr": Config.lr_backbone,
            "weight_decay": 0.0,
        },
        # Head / Structural parameters (fc, svd_norm, etc.)
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" not in n and not any(nd in n for nd in no_decay)
            ],
            "lr": Config.lr_head,
            "weight_decay": Config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" not in n and any(nd in n for nd in no_decay)
            ],
            "lr": Config.lr_head,
            "weight_decay": 0.0,
        },
    ]
    return optimizer_parameters
