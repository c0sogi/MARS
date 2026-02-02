import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import ModelConfig
from library.utils import AverageMeter
from library.model import AWP


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    """
    Trains the model for one epoch using the configured strategy.

    Features:
    - Gradient Accumulation
    - Adversarial Weight Perturbation (AWP) if enabled and epoch threshold met
    - Gradient Clipping
    """
    model.train()

    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    # Initialize AWP if configured and epoch condition is met
    use_awp = False
    if ModelConfig.use_awp and epoch >= ModelConfig.awp_start_epoch:
        use_awp = True
        awp = AWP(
            model, optimizer, adv_lr=ModelConfig.awp_lr, adv_eps=ModelConfig.awp_eps
        )

    dataset_size = len(dataloader.dataset)

    for step, data in enumerate(dataloader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        targets = data["target"].to(device)

        batch_size = input_ids.size(0)

        # =========================
        # Standard Forward Pass
        # =========================
        logits = model(input_ids, attention_mask)
        loss = criterion(logits.view(-1), targets)

        # Scale loss for gradient accumulation
        loss = loss / ModelConfig.accumulation_steps

        # Standard Backward Pass
        loss.backward()

        # Update loss meter (rescale to log true loss)
        losses.update(loss.item() * ModelConfig.accumulation_steps, batch_size)

        # =========================
        # Optimization Step
        # =========================
        if (step + 1) % ModelConfig.accumulation_steps == 0 or (step + 1) == len(
            dataloader
        ):

            # -------------------------
            # AWP Step (if enabled)
            # -------------------------
            if use_awp:
                # 1. Perturb weights based on accumulated gradients
                awp.attack()

                # 2. Forward pass with perturbed weights
                adv_logits = model(input_ids, attention_mask)
                adv_loss = criterion(adv_logits.view(-1), targets)

                # 3. Backward pass for adversarial loss
                adv_loss = adv_loss / ModelConfig.accumulation_steps
                adv_loss.backward()

                # 4. Restore original weights
                awp.restore()

            # -------------------------
            # Weight Update
            # -------------------------
            # Clip gradients to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), ModelConfig.max_grad_norm)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()

    return losses.avg


def valid_fn(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Returns:
        avg_loss (float): The average validation loss.
        auc_score (float): The Area Under the ROC Curve score.
    """
    model.eval()

    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    preds = []
    actuals = []

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            targets = data["target"].to(device)

            batch_size = input_ids.size(0)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits.view(-1), targets)

            losses.update(loss.item(), batch_size)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits.view(-1))

            preds.append(probs.detach().cpu().numpy())
            actuals.append(targets.detach().cpu().numpy())

    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)

    auc_score = roc_auc_score(actuals, preds)

    return losses.avg, auc_score


def inference_fn(model, dataloader, device):
    """
    Generates predictions for the test set.

    Returns:
        preds (np.array): Array of predicted probabilities.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits.view(-1))

            preds.append(probs.detach().cpu().numpy())

    preds = np.concatenate(preds)
    return preds
