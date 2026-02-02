import os
import gc
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForMaskedLM
from library.config import Config
from library.utils import AverageMeter, timeSince, get_logger


class AWP:
    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=1,
        adv_eps=0.2,
        start_epoch=0,
        scaler=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.scaler = scaler
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, inputs, labels, attention_mask, criterion, epoch):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save()
        self._attack_step()

        # Forward with perturbed weights
        # We assume mixed precision is handled by the caller or scaler if present
        if self.scaler:
            with torch.cuda.amp.autocast():
                y_preds = self.model(inputs, attention_mask)
                adv_loss = criterion(y_preds, labels)
            self.scaler.scale(adv_loss).backward()
        else:
            y_preds = self.model(inputs, attention_mask)
            adv_loss = criterion(y_preds, labels)
            adv_loss.backward()

        self._restore()
        return adv_loss

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


def compute_kl_loss(p, q):
    """
    Computes the symmetric KL divergence between two sets of logits p and q.
    Used for R-Drop.
    """
    p_loss = F.kl_div(F.logsigmoid(p), torch.sigmoid(q), reduction="none")
    q_loss = F.kl_div(F.logsigmoid(q), torch.sigmoid(p), reduction="none")

    # Sum over classes (dim 1), mean over batch (dim 0)
    p_loss = p_loss.sum(dim=1).mean()
    q_loss = q_loss.sum(dim=1).mean()

    loss = (p_loss + q_loss) / 2
    return loss


def train_fn(
    train_loader, model, criterion, optimizer, epoch, scheduler, device, config
):
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    losses = AverageMeter()
    start = time.time()

    # Initialize AWP
    awp = None
    if config.use_awp:
        awp = AWP(
            model,
            optimizer,
            adv_lr=config.awp_lr,
            adv_eps=config.awp_eps,
            start_epoch=config.awp_start_epoch,
            scaler=scaler,
        )

    for step, batch in enumerate(train_loader):
        inputs = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        batch_size = labels.size(0)

        with torch.cuda.amp.autocast():
            y_preds = model(inputs, mask)
            loss = criterion(y_preds, labels)

        losses.update(loss.item(), batch_size)

        # Backward
        scaler.scale(loss).backward()

        # AWP Attack
        if awp is not None:
            awp.attack_backward(inputs, labels, mask, criterion, epoch)

        # Gradient Clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.max_grad_norm
        )

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        if step % config.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                f"Epoch: [{epoch + 1}][{step}/{len(train_loader)}] "
                f"Elapsed: {timeSince(start, float(step + 1) / len(train_loader))} "
                f"Loss: {losses.avg:.4f} "
                f"Grad: {grad_norm:.4f} "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

    return losses.avg


def train_fn_rdrop(
    train_loader, model, criterion, optimizer, epoch, scheduler, device, config
):
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    losses = AverageMeter()
    start = time.time()

    awp = None
    if config.use_awp:
        awp = AWP(
            model,
            optimizer,
            adv_lr=config.awp_lr,
            adv_eps=config.awp_eps,
            start_epoch=config.awp_start_epoch,
            scaler=scaler,
        )

    for step, batch in enumerate(train_loader):
        inputs = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        batch_size = labels.size(0)

        with torch.cuda.amp.autocast():
            # Pass 1
            y_preds1 = model(inputs, mask)
            # Pass 2
            y_preds2 = model(inputs, mask)

            # Task Loss (Average of both passes)
            loss_task = 0.5 * (
                criterion(y_preds1, labels) + criterion(y_preds2, labels)
            )

            # KL Loss (Consistency)
            loss_kl = compute_kl_loss(y_preds1, y_preds2)

            loss = loss_task + config.rdrop_alpha * loss_kl

        losses.update(loss.item(), batch_size)

        scaler.scale(loss).backward()

        # AWP for R-Drop (Usually applied on the task loss or total loss)
        # Here we apply it simply using the first pass prediction logic or re-computing
        # For simplicity and speed in this complex loop, we often skip AWP or apply it standardly.
        # Given the constraints, we will apply AWP if enabled, but standard AWP expects one forward.
        # We will adapt AWP to use the first pass for perturbation calculation.
        if awp is not None and epoch >= config.awp_start_epoch:
            awp._save()
            awp._attack_step()

            with torch.cuda.amp.autocast():
                # Re-compute forward on perturbed weights (just one pass for efficiency)
                y_preds_adv = model(inputs, mask)
                loss_adv = criterion(y_preds_adv, labels)
                # We could also add KL here but it doubles cost again. Just task loss robustness is usually enough.

            scaler.scale(loss_adv).backward()
            awp._restore()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.max_grad_norm
        )

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        if step % config.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                f"Epoch: [{epoch + 1}][{step}/{len(train_loader)}] "
                f"Elapsed: {timeSince(start, float(step + 1) / len(train_loader))} "
                f"Loss: {losses.avg:.4f} "
                f"Grad: {grad_norm:.4f} "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

    return losses.avg


def valid_fn(val_loader, model, criterion, device, config):
    model.eval()
    losses = AverageMeter()
    preds = []
    start = time.time()

    for step, batch in enumerate(val_loader):
        inputs = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        batch_size = labels.size(0)

        with torch.no_grad():
            y_preds = model(inputs, mask)
            loss = criterion(y_preds, labels)

        losses.update(loss.item(), batch_size)
        preds.append(y_preds.sigmoid().to("cpu").numpy())

        if step % config.print_freq == 0 or step == (len(val_loader) - 1):
            print(
                f"EVAL: [{step}/{len(val_loader)}] "
                f"Elapsed: {timeSince(start, float(step + 1) / len(val_loader))} "
                f"Loss: {losses.avg:.4f}"
            )

    predictions = np.concatenate(preds)
    return losses.avg, predictions


def inference_fn(test_loader, model, device):
    model.eval()
    preds = []
    start = time.time()

    for step, batch in enumerate(test_loader):
        inputs = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            y_preds = model(inputs, mask)

        preds.append(y_preds.sigmoid().to("cpu").numpy())

        if step % 100 == 0 or step == (len(test_loader) - 1):
            print(
                f"INFERENCE: [{step}/{len(test_loader)}] "
                f"Elapsed: {timeSince(start, float(step + 1) / len(test_loader))}"
            )

    predictions = np.concatenate(preds)
    return predictions


def run_mlm(train_loader, model, optimizer, device, config, scheduler=None):
    """
    Runs Masked Language Modeling (DAPT).
    Note: 'model' here should be an AutoModelForMaskedLM, not the CustomModel.
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    losses = AverageMeter()
    start = time.time()

    print(f"Starting DAPT for {config.dapt_epochs} epochs...")

    for epoch in range(config.dapt_epochs):
        for step, batch in enumerate(train_loader):
            # MLM inputs
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            batch_size = input_ids.size(0)

            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss

            losses.update(loss.item(), batch_size)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

            if step % config.print_freq == 0 or step == (len(train_loader) - 1):
                print(
                    f"DAPT Epoch: [{epoch + 1}][{step}/{len(train_loader)}] "
                    f"Elapsed: {timeSince(start, float(step + 1) / len(train_loader))} "
                    f"Loss: {losses.avg:.4f}"
                )

    # Save the DAPT model
    print(f"Saving DAPT model to {config.dapt_model_path}...")
    model.save_pretrained(config.dapt_model_path)
    # Tokenizer is usually saved alongside, but we assume the caller handles tokenizer saving
    # or it's already saved. The model weights are the critical part here.
