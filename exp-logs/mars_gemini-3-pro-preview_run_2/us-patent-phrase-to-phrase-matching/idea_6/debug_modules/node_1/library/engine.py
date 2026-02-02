import time
import numpy as np
import torch
import torch.nn as nn
from library.config import CFG
from library.utils import get_score


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


def train_fn(train_loader, model, optimizer, epoch, scheduler, device, awp=None):
    """
    Performs one epoch of training with optional Adversarial Weight Perturbation (AWP).
    """
    model.train()
    scaler = torch.amp.GradScaler("cuda")
    losses = AverageMeter()
    start = time.time()

    # Label smoothing helps preventing overfitting on the specific discrete labels
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for step, inputs in enumerate(train_loader):
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        structural_features = inputs["structural_features"].to(device)
        labels = inputs["label"].to(device)

        # Map float scores 0.0, 0.25, 0.5, 0.75, 1.0 to indices 0, 1, 2, 3, 4
        label_indices = (labels * 4).round().long()
        batch_size = labels.size(0)

        # --- Standard Forward Pass ---
        with torch.amp.autocast("cuda"):
            y_preds = model(input_ids, attention_mask, structural_features)
            loss = criterion(y_preds, label_indices)

        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps

        scaler.scale(loss).backward()

        # --- Adversarial Weight Perturbation (AWP) ---
        # Only apply AWP after the warmup period defined in CFG
        if awp is not None and epoch >= CFG.awp_start_epoch:
            # 1. Perturb weights based on gradients from the standard backward pass
            awp.attack_step()

            # 2. Forward pass with perturbed weights
            with torch.amp.autocast("cuda"):
                y_preds_adv = model(input_ids, attention_mask, structural_features)
                loss_adv = criterion(y_preds_adv, label_indices)

            if CFG.gradient_accumulation_steps > 1:
                loss_adv = loss_adv / CFG.gradient_accumulation_steps

            # 3. Backward pass to accumulate gradients from adversarial loss
            scaler.scale(loss_adv).backward()

            # 4. Restore original weights
            awp._restore()

        # --- Optimization Step ---
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        losses.update(loss.item() * CFG.gradient_accumulation_steps, batch_size)

        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                f"Epoch: [{epoch + 1}][{step}/{len(train_loader)}] "
                f"Elapsed {time.time() - start:.2f}s "
                f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

    return losses.avg


def valid_fn(valid_loader, model, device):
    """
    Performs validation inference.
    Calculates the Expected Value of the predicted distribution to generate continuous scores.
    """
    model.eval()
    preds = []
    labels_list = []
    losses = AverageMeter()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    start = time.time()

    # Tensor for calculating expected value: [0.0, 0.25, 0.5, 0.75, 1.0]
    score_values = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]).to(device)

    with torch.no_grad():
        for step, inputs in enumerate(valid_loader):
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            structural_features = inputs["structural_features"].to(device)
            labels = inputs["label"].to(device)

            label_indices = (labels * 4).round().long()
            batch_size = labels.size(0)

            with torch.amp.autocast("cuda"):
                y_preds = model(input_ids, attention_mask, structural_features)
                loss = criterion(y_preds, label_indices)

            losses.update(loss.item(), batch_size)

            # --- Prediction Strategy: Expected Value ---
            # 1. Softmax to get probabilities for each class
            probs = torch.softmax(y_preds, dim=1)

            # 2. Dot product with score values to get expected score
            # Shape: (batch_size, 5) * (5,) -> sum -> (batch_size,)
            expected_scores = torch.sum(probs * score_values, dim=1)

            preds.append(expected_scores.to("cpu").numpy())
            labels_list.append(labels.to("cpu").numpy())

            if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
                print(
                    f"EVAL: [{step}/{len(valid_loader)}] "
                    f"Elapsed {time.time() - start:.2f}s "
                    f"Loss: {losses.val:.4f}({losses.avg:.4f})"
                )

    predictions = np.concatenate(preds)
    ground_truth = np.concatenate(labels_list)

    # Calculate Pearson Correlation
    score = get_score(ground_truth, predictions)

    return score, losses.avg, predictions
