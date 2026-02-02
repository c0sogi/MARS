import os
import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    get_scheduler,
)

from library.config import Config
from library.utils import JigsawMetric, set_seed
from library.losses import JigsawLoss
from library.model import AWP, ToxicityModel
from library.data import MLMDataset


def train_fn(
    model, data_loader, optimizer, scheduler, loss_fn, device, awp=None, epoch=0
):
    """
    Executes one training epoch.
    Includes logic for Adversarial Weight Perturbation (AWP) if provided.
    """
    model.train()
    scaler = GradScaler()
    losses = {"total": 0.0, "toxicity": 0.0, "rank": 0.0, "aux": 0.0}

    start_time = time.time()

    # Gradient accumulation setup
    accum_steps = Config.accumulate_grad_batches

    for step, batch in enumerate(data_loader):
        # Move inputs to device
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        targets = batch["target"].to(device, non_blocking=True).view(-1, 1)
        # identity_target and aux_target need to be combined for the loss function
        # which expects a single tensor for all auxiliary tasks
        identity_targets = batch["identity_target"].to(device, non_blocking=True)
        aux_targets = batch["aux_target"].to(device, non_blocking=True)
        combined_aux_targets = torch.cat([identity_targets, aux_targets], dim=1)

        weights = batch["weight"].to(device, non_blocking=True).view(-1, 1)

        # Mixed Precision Forward Pass
        with autocast():
            tox_logits, ident_logits, aux_logits = model(input_ids, attention_mask)
            combined_aux_logits = torch.cat([ident_logits, aux_logits], dim=1)

            loss, loss_dict = loss_fn(
                toxicity_logits=tox_logits,
                toxicity_targets=targets,
                aux_logits=combined_aux_logits,
                aux_targets=combined_aux_targets,
                sample_weights=weights,
            )

            # Normalize loss for gradient accumulation
            loss = loss / accum_steps

        # Backward Pass
        scaler.scale(loss).backward()

        # Adversarial Weight Perturbation (AWP)
        # AWP requires a second forward/backward pass with perturbed weights
        if awp is not None and epoch >= Config.awp_start_epoch:
            # Construct inputs dict for AWP's internal forward call
            awp_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "target": targets,
                "identity_target": identity_targets,
                "aux_target": aux_targets,
                "weight": weights,
            }
            # attack_backward handles the save -> attack -> forward -> backward -> restore cycle
            # Note: We pass the unscaled loss function; AWP handles scaling internally if provided
            awp.attack_backward(awp_inputs, loss_fn, epoch)

        # Optimizer Step (with accumulation)
        if (step + 1) % accum_steps == 0:
            # Unscale before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        # Logging
        losses["total"] += loss_dict["loss_total"]
        losses["toxicity"] += loss_dict["loss_toxicity"]
        losses["rank"] += loss_dict["loss_rank"]
        losses["aux"] += loss_dict["loss_aux"]

    # Average losses
    num_steps = len(data_loader)
    for k in losses:
        losses[k] /= num_steps

    return losses


def eval_fn(model, data_loader, loss_fn, device, df_valid):
    """
    Evaluates the model on the validation set.
    Computes Loss and Jigsaw Metrics.
    """
    model.eval()
    losses = {"total": 0.0, "toxicity": 0.0, "rank": 0.0, "aux": 0.0}

    preds_toxicity = []

    # We don't strictly need identity preds for validation metric,
    # but we compute loss which requires them.

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            targets = batch["target"].to(device, non_blocking=True).view(-1, 1)
            identity_targets = batch["identity_target"].to(device, non_blocking=True)
            aux_targets = batch["aux_target"].to(device, non_blocking=True)
            combined_aux_targets = torch.cat([identity_targets, aux_targets], dim=1)
            weights = batch["weight"].to(device, non_blocking=True).view(-1, 1)

            tox_logits, ident_logits, aux_logits = model(input_ids, attention_mask)
            combined_aux_logits = torch.cat([ident_logits, aux_logits], dim=1)

            loss, loss_dict = loss_fn(
                toxicity_logits=tox_logits,
                toxicity_targets=targets,
                aux_logits=combined_aux_logits,
                aux_targets=combined_aux_targets,
                sample_weights=weights,
            )

            # Accumulate Loss
            losses["total"] += loss_dict["loss_total"]
            losses["toxicity"] += loss_dict["loss_toxicity"]
            losses["rank"] += loss_dict["loss_rank"]
            losses["aux"] += loss_dict["loss_aux"]

            # Store Predictions (Sigmoid applied)
            preds_toxicity.append(torch.sigmoid(tox_logits).cpu().numpy())

    # Average losses
    num_steps = len(data_loader)
    for k in losses:
        losses[k] /= num_steps

    # Concatenate predictions
    preds_toxicity = np.concatenate(preds_toxicity).flatten()

    # Compute Jigsaw Metrics
    metric_calculator = JigsawMetric()
    metrics = metric_calculator.compute(df_valid, preds_toxicity)

    return losses, metrics, preds_toxicity


def inference_fn(model, data_loader, device):
    """
    Runs inference to generate predictions.
    Used for both 'Mining' (Stage 2) and 'Submission' (Final).
    Returns toxicity probabilities and identity probabilities.
    """
    model.eval()
    preds_toxicity = []
    preds_identity = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            tox_logits, ident_logits, _ = model(input_ids, attention_mask)

            preds_toxicity.append(torch.sigmoid(tox_logits).cpu().numpy())
            preds_identity.append(torch.sigmoid(ident_logits).cpu().numpy())

    preds_toxicity = np.concatenate(preds_toxicity).flatten()
    preds_identity = np.concatenate(preds_identity, axis=0)

    return {"toxicity": preds_toxicity, "identity": preds_identity}


def run_mlm(train_texts, val_texts, tokenizer, device):
    """
    Stage 1: Domain-Adaptive Pretraining (MLM).
    Adapts the backbone to the competition corpus.
    """
    print("Starting Domain-Adaptive Pretraining (MLM)...")

    # Configuration
    model_name = Config.model_name
    epochs = Config.dapt_epochs
    batch_size = Config.train_batch_size  # Reuse batch size
    lr = Config.dapt_lr

    # Initialize Model for MLM
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)
    model.train()

    # Prepare Data
    # Combine texts if val provided, or just use train (usually we use all available text)
    all_texts = train_texts + (val_texts if val_texts else [])
    dataset = MLMDataset(all_texts, tokenizer, Config.max_len)

    # Data Collator handles masking
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=Config.dapt_mask_probability
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=Config.weight_decay)

    scaler = GradScaler()

    for epoch in range(epochs):
        total_loss = 0
        start_time = time.time()

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(
            f"MLM Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | Time: {time.time() - start_time:.0f}s"
        )

    # Save the adapted encoder
    # We save the 'roberta' or 'deberta' part so it can be loaded into ToxicityModel
    # DebertaV3 uses 'deberta' attribute usually, but save_pretrained saves the whole config
    save_path = os.path.join(Config.working_dir, "dapt_model")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"DAPT Complete. Model saved to {save_path}")

    # Cleanup
    del model, optimizer, loader, dataset
    gc.collect()
    torch.cuda.empty_cache()

    return save_path
