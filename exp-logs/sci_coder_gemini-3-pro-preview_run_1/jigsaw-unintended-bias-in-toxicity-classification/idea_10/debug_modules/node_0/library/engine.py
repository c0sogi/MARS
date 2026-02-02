import torch
import torch.nn as nn
import numpy as np
import time
import gc
from library.config import CFG
from library.utils import JigsawMetrics, AWP, EMA
from library.losses import JigsawLoss


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


def train_mlm(model, train_loader, optimizer, scheduler, device, epoch):
    """
    Stage 1: Domain-Adaptive Pretraining (MLM)
    """
    model.train()
    losses = AverageMeter()
    start = time.time()

    # MLM models (AutoModelForMaskedLM) compute loss internally if labels are provided
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        batch_size = input_ids.size(0)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps

        loss.backward()

        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        losses.update(loss.item() * CFG.gradient_accumulation_steps, batch_size)

        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                f"Epoch: [{epoch + 1}][{step}/{len(train_loader)}] "
                f"Elapsed {time.time() - start:.1f}s "
                f"Loss: {losses.val:.4f} "
                f"Loss Avg: {losses.avg:.4f} "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

    return losses.avg


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    epoch,
    stage,
    awp=None,
    ema=None,
):
    """
    Stage 2 & 3: Classification Training
    Handles both 'general' and 'robust' stages.
    """
    model.train()
    losses = AverageMeter()
    start = time.time()

    for step, batch in enumerate(train_loader):
        # Move inputs to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target = batch["target"].to(device).unsqueeze(1)
        identities = batch["identities"].to(device)
        attack = batch["attack"].to(device).unsqueeze(1)
        sample_weights = batch["sample_weights"].to(device).unsqueeze(1)

        batch_size = input_ids.size(0)

        targets_dict = {
            "target": target,
            "identities": identities,
            "attack": attack,
            "sample_weights": sample_weights,
        }

        # --- Forward Pass ---
        outputs = model(input_ids, attention_mask)
        loss, _ = criterion(outputs, targets_dict, stage=stage)

        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps

        # --- Backward Pass ---
        loss.backward()

        # --- Robust Optimization (AWP) ---
        # Only applied in Stage 3 ('robust') if AWP is initialized
        if stage == "robust" and awp is not None:
            # 1. Perturb weights based on current gradients
            awp.attack_step()

            # 2. Clear clean gradients to optimize for the adversarial point
            # (Standard AWP strategy: minimize max loss)
            model.zero_grad()

            # 3. Forward pass with perturbed weights
            adv_outputs = model(input_ids, attention_mask)
            adv_loss, _ = criterion(adv_outputs, targets_dict, stage=stage)

            if CFG.gradient_accumulation_steps > 1:
                adv_loss = adv_loss / CFG.gradient_accumulation_steps

            # 4. Backward pass to accumulate gradients from adversarial point
            adv_loss.backward()

            # 5. Restore original weights (gradients remain)
            awp.restore()

        # --- Optimizer Step ---
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

            # Update EMA
            if ema is not None:
                ema.update()

        losses.update(loss.item() * CFG.gradient_accumulation_steps, batch_size)

        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                f"Epoch: [{epoch + 1}][{step}/{len(train_loader)}] "
                f"Stage: {stage} "
                f"Elapsed {time.time() - start:.1f}s "
                f"Loss: {losses.val:.4f} "
                f"Loss Avg: {losses.avg:.4f} "
                f"LR: {optimizer.param_groups[0]['lr']:.8f}"
            )

    return losses.avg


def valid_epoch(model, valid_loader, criterion, device, ema=None):
    """
    Validation Loop.
    Uses EMA weights if provided. Calculates Jigsaw Metrics.
    """
    if ema is not None:
        ema.apply_shadow()

    model.eval()
    losses = AverageMeter()
    preds = []

    # Store targets for metric calculation
    # We need to reconstruct the dataframe-like structure for JigsawMetrics
    target_list = []
    identity_list = []

    start = time.time()

    with torch.no_grad():
        for step, batch in enumerate(valid_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["target"].to(device).unsqueeze(1)
            identities = batch["identities"].to(device)
            attack = batch["attack"].to(device).unsqueeze(1)
            # Validation doesn't use sample weights usually, but we pass ones just in case
            sample_weights = torch.ones_like(target)

            batch_size = input_ids.size(0)

            targets_dict = {
                "target": target,
                "identities": identities,
                "attack": attack,
                "sample_weights": sample_weights,
            }

            outputs = model(input_ids, attention_mask)

            # We use 'general' stage for validation loss monitoring (no ranking loss)
            loss, _ = criterion(outputs, targets_dict, stage="general")

            losses.update(loss.item(), batch_size)

            # Collect predictions (sigmoid applied here or in metric?
            # BCEWithLogitsLoss takes logits. Metric usually takes probabilities.
            # JigsawMetrics.compute_auc takes raw scores usually, but let's apply sigmoid to be safe/standard)
            batch_preds = torch.sigmoid(outputs["logits"]).detach().cpu().numpy()
            preds.append(batch_preds)

            target_list.append(target.cpu().numpy())
            identity_list.append(identities.cpu().numpy())

            if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
                print(
                    f"EVAL: [{step}/{len(valid_loader)}] "
                    f"Elapsed {time.time() - start:.1f}s "
                    f"Loss: {losses.val:.4f} "
                    f"Loss Avg: {losses.avg:.4f}"
                )

    if ema is not None:
        ema.restore()

    # --- Metric Calculation ---
    predictions = np.concatenate(preds)
    targets = np.concatenate(target_list)
    identities = np.concatenate(identity_list)

    # Reconstruct DataFrame for JigsawMetrics
    # Columns: target, male, female, ...
    data_dict = {CFG.target_col: targets.flatten()}
    for i, col in enumerate(CFG.identity_cols):
        data_dict[col] = identities[:, i]

    val_df = import_pandas_for_eval(data_dict)

    metrics_calculator = JigsawMetrics()
    final_score, overall_auc, sub_auc, bpsn_auc, bnsp_auc = (
        metrics_calculator.get_final_metric(val_df, predictions)
    )

    print(f"Validation Result:")
    print(f"Score: {final_score}")
    print(f"Overall AUC: {overall_auc}")
    print(f"Subgroup AUC: {sub_auc}")
    print(f"BPSN AUC: {bpsn_auc}")
    print(f"BNSP AUC: {bnsp_auc}")

    return losses.avg, final_score


def inference(model, test_loader, device, ema=None):
    """
    Inference Loop for Test Set.
    """
    if ema is not None:
        ema.apply_shadow()

    model.eval()
    preds = []
    start = time.time()

    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask)

            # Apply sigmoid to get probabilities
            batch_preds = torch.sigmoid(outputs["logits"]).detach().cpu().numpy()
            preds.append(batch_preds)

            if step % CFG.print_freq == 0 or step == (len(test_loader) - 1):
                print(
                    f"TEST: [{step}/{len(test_loader)}] "
                    f"Elapsed {time.time() - start:.1f}s"
                )

    if ema is not None:
        ema.restore()

    predictions = np.concatenate(preds)
    return predictions


def import_pandas_for_eval(data_dict):
    import pandas as pd

    return pd.DataFrame(data_dict)
