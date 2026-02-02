import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config
from library.utils import AverageMeter, compute_pearson
from library.awp import AWP


def criterion(outputs, targets):
    """
    Computes KL Divergence Loss.
    Args:
        outputs: Logits from the model (Batch, Num_Classes)
        targets: Soft target probabilities (Batch, Num_Classes)
    """
    # KLDivLoss expects log_probabilities for input
    log_probs = F.log_softmax(outputs, dim=1)
    loss = nn.KLDivLoss(reduction="batchmean")(log_probs, targets)
    return loss


def get_expected_score(logits):
    """
    Converts logits to a scalar score using Expected Value.
    Classes correspond to [0.0, 0.25, 0.5, 0.75, 1.0].
    """
    probs = F.softmax(logits, dim=1)  # (Batch, 5)
    # Class values tensor
    classes = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=logits.device)
    # Expected value: sum(p_i * x_i)
    preds = torch.sum(probs * classes, dim=1)
    return preds


def train_fn(train_loader, model, optimizer, epoch, scheduler, device, awp=None):
    """
    Training loop for one epoch.
    Includes Gradient Accumulation and Adversarial Weight Perturbation (AWP).
    """
    model.train()
    losses = AverageMeter()

    # Reset gradients at the start of the epoch
    optimizer.zero_grad()

    num_steps = len(train_loader)

    for step, batch in enumerate(train_loader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        structural_features = batch["structural_features"].to(device)
        labels = batch["labels"].to(device)

        batch_size = input_ids.size(0)

        # --- 1. Clean Forward Pass ---
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            structural_features=structural_features,
        )

        loss = criterion(logits, labels)

        # Normalize loss for gradient accumulation
        loss = loss / Config.gradient_accumulation_steps

        # Backward pass to compute gradients
        loss.backward()

        # --- 2. Adversarial Pass (AWP) ---
        if awp is not None and epoch >= Config.awp_start_epoch:
            # Perturb weights based on current gradients
            awp.attack_step()

            # Forward pass with perturbed weights
            adv_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                structural_features=structural_features,
            )

            adv_loss = criterion(adv_logits, labels)
            adv_loss = adv_loss / Config.gradient_accumulation_steps

            # We want to optimize the adversarial loss.
            # Clear the "clean" gradients and replace with adversarial gradients.
            optimizer.zero_grad()
            adv_loss.backward()

            # Restore original weights
            awp.restore()

        # Track loss (use the clean loss for logging)
        losses.update(loss.item() * Config.gradient_accumulation_steps, batch_size)

        # --- 3. Optimizer Step ---
        if (step + 1) % Config.gradient_accumulation_steps == 0 or (
            step + 1
        ) == num_steps:
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()

    return losses.avg


def eval_fn(data_loader, model, device):
    """
    Evaluation loop.
    Computes Pearson Correlation on the validation/test set.
    """
    model.eval()
    losses = AverageMeter()

    final_preds = []
    final_targets = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            structural_features = batch["structural_features"].to(device)

            # Labels might not exist in test set, but exist in val
            labels = None
            if "labels" in batch:
                labels = batch["labels"].to(device)

            # Raw scores for metric calculation
            scores = None
            if "scores" in batch:
                scores = batch["scores"].numpy()

            batch_size = input_ids.size(0)

            # Forward Pass
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                structural_features=structural_features,
            )

            # Compute Loss if labels available
            if labels is not None:
                loss = criterion(logits, labels)
                losses.update(loss.item(), batch_size)

            # Generate Predictions (Expected Value)
            preds = get_expected_score(logits)

            final_preds.extend(preds.cpu().numpy())
            if scores is not None:
                final_targets.extend(scores)

    # Compute Metric
    score = 0.0
    if len(final_targets) > 0:
        score = compute_pearson(final_targets, final_preds)

    return losses.avg, np.array(final_preds), score
