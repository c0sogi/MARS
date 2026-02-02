import os
import time
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from library.config import CFG
from library.utils import get_logger, compute_qwk
from library.awp import AWP

logger = get_logger(os.path.join(CFG.output_dir, "train.log"))


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


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    """
    Configures layer-wise learning rate decay (LLRD) for the model.
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = []

    # Get number of layers from config if available, default to 24 for large
    num_layers = 24
    if hasattr(model.backbone.config, "num_hidden_layers"):
        num_layers = model.backbone.config.num_hidden_layers

    # Initialize layer groups
    # Group 0: Embeddings
    # Group 1..N: Encoder Layers
    # Group N+1: Head

    # 1. Embeddings
    embedding_params = []
    for n, p in model.backbone.embeddings.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n for nd in no_decay):
            embedding_params.append(
                {
                    "params": [p],
                    "weight_decay": 0.0,
                    "lr": encoder_lr * (CFG.llrd_decay**num_layers),
                }
            )
        else:
            embedding_params.append(
                {
                    "params": [p],
                    "weight_decay": weight_decay,
                    "lr": encoder_lr * (CFG.llrd_decay**num_layers),
                }
            )
    optimizer_parameters.extend(embedding_params)

    # 2. Encoder Layers
    for layer_idx in range(num_layers):
        layer_params = []
        # Access the specific layer
        layer = model.backbone.encoder.layer[layer_idx]
        lr = encoder_lr * (CFG.llrd_decay ** (num_layers - layer_idx - 1))

        for n, p in layer.named_parameters():
            if not p.requires_grad:
                continue
            if any(nd in n for nd in no_decay):
                layer_params.append({"params": [p], "weight_decay": 0.0, "lr": lr})
            else:
                layer_params.append(
                    {"params": [p], "weight_decay": weight_decay, "lr": lr}
                )
        optimizer_parameters.extend(layer_params)

    # 3. Head / Decoder / Pooler
    head_params = []
    head_names = ["pool", "fc", "rel_embeddings", "LayerNorm"]
    # Note: rel_embeddings might be in backbone but outside encoder in some DeBERTa impls,
    # but usually handled in embeddings or encoder. We focus on custom head layers here.

    # Iterate all params and catch those not in embeddings or encoder layers
    # A simpler way for the head is to explicitly grab model.pool and model.fc

    for module in [model.pool, model.fc]:
        for n, p in module.named_parameters():
            if not p.requires_grad:
                continue
            if any(nd in n for nd in no_decay):
                head_params.append(
                    {"params": [p], "weight_decay": 0.0, "lr": decoder_lr}
                )
            else:
                head_params.append(
                    {"params": [p], "weight_decay": weight_decay, "lr": decoder_lr}
                )

    optimizer_parameters.extend(head_params)

    return optimizer_parameters


def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    """
    Performs one epoch of training.
    """
    model.train()
    scaler = torch.amp.GradScaler("cuda")
    losses = AverageMeter()
    start = time.time()

    # Initialize AWP if applicable
    awp = None
    if CFG.use_awp and epoch >= CFG.awp_start_epoch:
        awp = AWP(model, optimizer, adv_lr=CFG.awp_lr, adv_eps=CFG.awp_eps)

    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        # Move inputs to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        chunk_mask = batch["chunk_mask"].to(device)
        labels = batch["labels"].to(device)

        batch_size = labels.size(0)

        # Mixed Precision Forward Pass
        with torch.amp.autocast("cuda"):
            output = model(input_ids, attention_mask, chunk_mask)
            logits = output["logits"].squeeze(-1)
            loss = criterion(logits, labels)

            if CFG.accum_iter > 1:
                loss = loss / CFG.accum_iter

        losses.update(loss.item() * CFG.accum_iter, batch_size)

        # Backward Pass
        scaler.scale(loss).backward()

        # AWP Attack
        if awp is not None:
            # Unscale gradients before AWP to get correct magnitudes?
            # AWP implementation usually handles raw gradients.
            # Standard AWP flow with scaler:
            # 1. scaler.unscale_(optimizer) (optional, but good for clipping)
            # 2. awp.attack()
            # 3. forward + backward
            # 4. awp.restore()

            # We preserve the scaler flow.
            # Note: AWP requires gradients to be available.

            # Attack
            scaler.unscale_(optimizer)  # Cite debug_lesson_8
            awp.attack()

            with torch.amp.autocast("cuda"):
                output_adv = model(input_ids, attention_mask, chunk_mask)
                logits_adv = output_adv["logits"].squeeze(-1)
                loss_adv = criterion(logits_adv, labels)
                if CFG.accum_iter > 1:
                    loss_adv = loss_adv / CFG.accum_iter

            # Backward on adversarial loss
            loss_adv.backward()

            # Restore original weights
            awp.restore()

        # Gradient Accumulation Step
        if (step + 1) % CFG.accum_iter == 0 or (step + 1) == len(train_loader):
            # Unscale if not already unscaled (if AWP ran, it was unscaled, but scaler handles double unscale safely usually or we re-check)
            # To be safe with PyTorch scaler, we just call unscale_ before clip
            if awp is None:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                f"Epoch: [{epoch+1}][{step}/{len(train_loader)}] "
                f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
                f'LR: {optimizer.param_groups[0]["lr"]:.8f} '
                f"Elapsed: {time.time() - start:.2f}s"
            )

    return losses.avg


def valid_fn(val_loader, model, criterion, device):
    """
    Performs validation.
    """
    model.eval()
    losses = AverageMeter()
    preds = []
    start = time.time()

    for step, batch in enumerate(val_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        chunk_mask = batch["chunk_mask"].to(device)
        labels = batch["labels"].to(device)
        batch_size = labels.size(0)

        with torch.no_grad():
            output = model(input_ids, attention_mask, chunk_mask)
            logits = output["logits"].squeeze(-1)
            loss = criterion(logits, labels)

        losses.update(loss.item(), batch_size)
        preds.append(logits.to("cpu").numpy())

    predictions = np.concatenate(preds)

    # Calculate Validation QWK
    # We assume the loader provides labels in order.
    # Reconstruct labels from loader for metric calculation to be safe
    all_labels = []
    for batch in val_loader:
        all_labels.append(batch["labels"].numpy())
    ground_truth = np.concatenate(all_labels)

    # Clip and Round for QWK
    val_preds_int = np.rint(np.clip(predictions, 1, 6)).astype(int)
    val_labels_int = ground_truth.astype(int)

    score = compute_qwk(val_labels_int, val_preds_int)

    print(
        f"EVAL: Loss: {losses.avg:.4f} QWK: {score:.5f} Elapsed: {time.time() - start:.2f}s"
    )

    return losses.avg, predictions


def extract_embeddings_fn(loader, model, device):
    """
    Extracts embeddings for Stacking.
    Returns:
        embeddings: numpy array of shape (n_samples, hidden_size)
        labels: numpy array of shape (n_samples,) or None
    """
    model.eval()
    embeddings_list = []
    labels_list = []

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        chunk_mask = batch["chunk_mask"].to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask, chunk_mask)
            emb = output["embedding"]

        embeddings_list.append(emb.to("cpu").numpy())

        if "labels" in batch:
            labels_list.append(batch["labels"].numpy())

    embeddings = np.concatenate(embeddings_list)
    labels = np.concatenate(labels_list) if len(labels_list) > 0 else None

    return embeddings, labels
