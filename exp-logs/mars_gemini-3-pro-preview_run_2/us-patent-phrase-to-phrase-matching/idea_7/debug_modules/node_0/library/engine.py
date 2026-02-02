import torch
import torch.nn as nn
import numpy as np
import time
from torch.cuda.amp import autocast, GradScaler
from library.config import Config
from library.utils import compute_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_fn(
    fold, train_loader, model, criterion, optimizer, epoch, scheduler, device, awp=None
):
    """
    Performs one epoch of training with Mixed Precision and AWP.
    """
    model.train()
    scaler = GradScaler()
    losses = AverageMeter()
    start = time.time()

    global_step = 0

    for step, batch in enumerate(train_loader):
        # Unpack batch and move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        structural_features = batch["structural_features"].to(device)
        labels = batch["labels"].to(device)
        batch_size = labels.size(0)

        # Mixed Precision Forward Pass
        with autocast():
            y_preds = model(input_ids, attention_mask, structural_features)
            loss = criterion(y_preds, labels)

        # Scale loss for gradient accumulation
        if Config.gradient_accumulation_steps > 1:
            loss = loss / Config.gradient_accumulation_steps

        # Record loss (scale back up for logging)
        losses.update(loss.item() * Config.gradient_accumulation_steps, batch_size)

        # Backward Pass (Clean)
        scaler.scale(loss).backward()

        # Adversarial Weight Perturbation (AWP)
        if awp is not None:
            # awp.perturb returns True only if the current epoch >= awp_start_epoch
            if awp.perturb(epoch):
                # Forward pass with perturbed weights
                with autocast():
                    y_preds_adv = model(input_ids, attention_mask, structural_features)
                    loss_adv = criterion(y_preds_adv, labels)
                    if Config.gradient_accumulation_steps > 1:
                        loss_adv = loss_adv / Config.gradient_accumulation_steps

                # Backward pass (Adversarial) - Accumulate gradients
                scaler.scale(loss_adv).backward()

                # Restore original weights
                awp.restore()

        # Optimizer Step (Gradient Accumulation)
        if (step + 1) % Config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

            global_step += 1

        if step % Config.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                f"Epoch: [{epoch}][{step}/{len(train_loader)}] "
                f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
                f"LR: {scheduler.get_last_lr()[0]:.8f} "
                f"Elapsed: {time.time() - start:.0f}s"
            )

    return losses.avg


def valid_fn(val_loader, model, criterion, device):
    """
    Performs validation/inference.
    Calculates Expected Value from classification probabilities for scoring.
    """
    model.eval()
    losses = AverageMeter()
    preds = []
    targets = []
    start = time.time()

    # Values for Expected Value calculation: 0.0, 0.25, 0.5, 0.75, 1.0
    score_values = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device)

    for step, batch in enumerate(val_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        structural_features = batch["structural_features"].to(device)
        labels = batch["labels"].to(device)
        raw_scores = batch["scores"].to(device)  # Float scores for metric calculation
        batch_size = labels.size(0)

        with torch.no_grad():
            y_preds = model(input_ids, attention_mask, structural_features)
            loss = criterion(y_preds, labels)

        losses.update(loss.item(), batch_size)

        # Calculate Expected Value from logits
        # 1. Softmax to get probabilities
        probs = torch.softmax(y_preds, dim=1)
        # 2. Weighted sum of score values (Expectation)
        pred_scores = torch.sum(probs * score_values, dim=1)

        preds.append(pred_scores.to("cpu").numpy())
        targets.append(raw_scores.to("cpu").numpy())

        if step % Config.print_freq == 0 or step == (len(val_loader) - 1):
            print(
                f"EVAL: [{step}/{len(val_loader)}] "
                f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
                f"Elapsed: {time.time() - start:.0f}s"
            )

    predictions = np.concatenate(preds)
    ground_truth = np.concatenate(targets)

    score = compute_score(ground_truth, predictions)

    return losses.avg, score, predictions
