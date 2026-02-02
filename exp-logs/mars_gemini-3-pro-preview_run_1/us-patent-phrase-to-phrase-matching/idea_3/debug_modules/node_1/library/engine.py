import time
import math
import numpy as np
import torch
import torch.nn as nn
from library.utils import AverageMeter, get_score
from library.awp import AWP


def asMinutes(s):
    """
    Converts seconds to a string format 'm m s s'.
    """
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    """
    Calculates elapsed and remaining time based on progress.
    """
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))


def train_fn(
    fold, train_loader, model, criterion, optimizer, epoch, scheduler, device, cfg
):
    """
    Performs one epoch of training.
    """
    model.train()
    scaler = torch.amp.GradScaler("cuda")
    losses = AverageMeter()
    start = time.time()

    # Initialize Adversarial Weight Perturbation
    awp = AWP(
        model,
        optimizer,
        adv_lr=cfg.awp_lr,
        adv_eps=cfg.awp_eps,
        start_epoch=cfg.awp_start_epoch,
        scaler=scaler,
    )

    for step, inputs in enumerate(train_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        labels = inputs["label"]
        batch_size = labels.size(0)

        # Mixed Precision Forward Pass (Clean)
        with torch.amp.autocast("cuda"):
            y_preds = model(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs.get("token_type_ids"),
            )
            loss = criterion(y_preds.view(-1), labels.view(-1))

        # Record unscaled loss for logging
        losses.update(loss.item(), batch_size)

        # Scale loss for gradient accumulation
        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps

        scaler.scale(loss).backward()

        # Perform updates only after accumulation steps
        if (step + 1) % cfg.gradient_accumulation_steps == 0:

            # AWP Attack Logic
            # Only apply AWP if enabled and after the start epoch
            if cfg.awp and epoch >= cfg.awp_start_epoch:
                # 1. Perturb weights based on current gradients
                awp.attack(epoch)

                # 2. Forward pass with perturbed weights
                with torch.amp.autocast("cuda"):
                    y_preds_adv = model(
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        inputs.get("token_type_ids"),
                    )
                    loss_adv = criterion(y_preds_adv.view(-1), labels.view(-1))
                    if cfg.gradient_accumulation_steps > 1:
                        loss_adv = loss_adv / cfg.gradient_accumulation_steps

                # 3. Clear clean gradients and compute adversarial gradients
                optimizer.zero_grad()
                scaler.scale(loss_adv).backward()

                # 4. Restore original weights for the update step
                awp._restore()

            # Gradient Clipping
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.max_grad_norm
            )

            # Optimizer Step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if cfg.batch_scheduler:
                scheduler.step()

        if step % cfg.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                "Epoch: [{0}][{1}/{2}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                "Grad: {grad_norm:.4f}  "
                "LR: {lr:.8f}  ".format(
                    epoch + 1,
                    step,
                    len(train_loader),
                    remain=timeSince(start, float(step + 1) / len(train_loader)),
                    loss=losses,
                    grad_norm=grad_norm if "grad_norm" in locals() else 0,
                    lr=scheduler.get_last_lr()[0],
                )
            )


def valid_fn(valid_loader, model, criterion, device, cfg):
    """
    Performs validation on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    preds = []
    labels_list = []
    start = time.time()

    for step, inputs in enumerate(valid_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        labels = inputs["label"]
        batch_size = labels.size(0)

        with torch.no_grad():
            y_preds = model(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs.get("token_type_ids"),
            )
            loss = criterion(y_preds.view(-1), labels.view(-1))

        losses.update(loss.item(), batch_size)
        preds.append(y_preds.cpu().numpy())
        labels_list.append(labels.cpu().numpy())

        if step % cfg.print_freq == 0 or step == (len(valid_loader) - 1):
            print(
                "EVAL: [{0}/{1}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) ".format(
                    step,
                    len(valid_loader),
                    remain=timeSince(start, float(step + 1) / len(valid_loader)),
                    loss=losses,
                )
            )

    predictions = np.concatenate(preds)
    ground_truth = np.concatenate(labels_list)

    # Calculate Pearson Correlation
    score = get_score(ground_truth, predictions)

    # Print full precision score as requested
    print(f"Validation Score: {score}")

    return losses.avg, predictions, score
