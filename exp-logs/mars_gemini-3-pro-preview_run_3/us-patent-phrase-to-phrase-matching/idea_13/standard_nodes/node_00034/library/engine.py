import time
import numpy as np
import torch
import torch.nn as nn
from library.config import Config
from library.training_utils import AWP, EMA


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
    fold,
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    scheduler,
    device,
    cfg,
    awp=None,
    ema=None,
):
    """
    Executes one training epoch.
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    losses = AverageMeter()

    num_steps = len(train_loader)

    for step, inputs in enumerate(train_loader):
        # Move inputs to device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        labels = inputs["labels"]
        batch_size = labels.size(0)

        # Forward Pass with AMP
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(inputs["input_ids"], inputs["attention_mask"])
            loss_dict = criterion(outputs, labels)
            loss = loss_dict["loss"]

        # Normalize loss for gradient accumulation
        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps

        # Backward Pass
        scaler.scale(loss).backward()

        # Adversarial Weight Perturbation (AWP)
        if awp is not None and epoch >= cfg.awp_start_epoch:
            # We do not unscale here to allow AWP to work on scaled gradients (scale invariant)
            awp.attack()

            with torch.cuda.amp.autocast(enabled=True):
                outputs_adv = model(inputs["input_ids"], inputs["attention_mask"])
                loss_dict_adv = criterion(outputs_adv, labels)
                loss_adv = loss_dict_adv["loss"]

            if cfg.gradient_accumulation_steps > 1:
                loss_adv = loss_adv / cfg.gradient_accumulation_steps

            scaler.scale(loss_adv).backward()
            awp.restore()

        # Optimizer Step
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

            # Exponential Moving Average (EMA) Update
            if ema is not None and epoch >= cfg.ema_start_epoch:
                ema.update()

        losses.update(loss_dict["loss"].item(), batch_size)

        if step % cfg.print_freq == 0 or step == (num_steps - 1):
            print(
                f"Epoch: [{epoch + 1}][{step}/{num_steps}] "
                f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
                f"LR: {optimizer.param_groups[0]['lr']:.8f}"
            )

    return losses.avg


def valid_fn(fold, valid_loader, model, criterion, device, cfg, ema=None):
    """
    Evaluates the model on the validation set.
    """
    # Apply EMA shadow weights for validation
    if ema is not None:
        ema.apply_shadow()
        print("Validating with EMA weights...")

    model.eval()
    losses = AverageMeter()
    preds = []
    targets = []

    for step, inputs in enumerate(valid_loader):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        labels = inputs["labels"]
        batch_size = labels.size(0)

        with torch.no_grad():
            outputs = model(inputs["input_ids"], inputs["attention_mask"])
            loss_dict = criterion(outputs, labels)

        losses.update(loss_dict["loss"].item(), batch_size)

        # Collect predictions (regression logits)
        preds.append(outputs["logits"].view(-1).detach().cpu().numpy())
        targets.append(labels.view(-1).detach().cpu().numpy())

    # Restore original weights
    if ema is not None:
        ema.restore()

    predictions = np.concatenate(preds)
    ground_truth = np.concatenate(targets)

    # Calculate Pearson Correlation
    # We use full precision for printing as requested
    pearson_score = np.corrcoef(predictions, ground_truth)[0, 1]

    print(f"Validation Fold {fold} - Loss: {losses.avg} Pearson: {pearson_score}")

    return losses.avg, pearson_score, predictions


def inference_fn(test_loader, model, device, cfg, ema=None):
    """
    Generates predictions for the test set.
    """
    # Apply EMA shadow weights for inference
    if ema is not None:
        ema.apply_shadow()
        print("Inference with EMA weights...")

    model.eval()
    preds = []
    ids = []

    for step, inputs in enumerate(test_loader):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        with torch.no_grad():
            outputs = model(inputs["input_ids"], inputs["attention_mask"])

        # Collect regression logits
        batch_preds = outputs["logits"].view(-1).detach().cpu().numpy()
        preds.append(batch_preds)
        ids.extend(inputs["id"])

    if ema is not None:
        ema.restore()

    predictions = np.concatenate(preds)

    # Clip predictions to valid range [0, 1]
    predictions = np.clip(predictions, 0, 1)

    return ids, predictions
