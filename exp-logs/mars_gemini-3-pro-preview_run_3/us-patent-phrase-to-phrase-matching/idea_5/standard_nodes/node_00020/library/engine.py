import os
import gc
import time
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from tqdm.auto import tqdm

from library.config import Config
from library.utils import get_logger, AverageMeter, get_score, seed_everything
from library.dataset import get_dapt_dataloader
from library.awp import AWP

logger = get_logger("engine")


def run_dapt(tokenizer):
    """
    Performs Domain-Adaptive Pre-training (DAPT) using Masked Language Modeling.
    Saves the pre-trained backbone to Config.dapt_model_path.
    """
    if not Config.use_dapt:
        logger.info("DAPT is disabled in Config. Skipping.")
        return

    save_path = Config.dapt_model_path
    if os.path.exists(save_path) and os.path.exists(
        os.path.join(save_path, "config.json")
    ):
        logger.info(f"DAPT model already exists at {save_path}. Skipping training.")
        return

    logger.info("Starting Domain-Adaptive Pre-training (DAPT)...")

    # Create DAPT directory
    os.makedirs(save_path, exist_ok=True)

    # Initialize Model for MLM
    device = Config.device
    config = AutoConfig.from_pretrained(Config.model_name)
    config.use_cache = False
    model = AutoModelForMaskedLM.from_pretrained(Config.model_name, config=config)
    model.gradient_checkpointing_enable()
    model.to(device)
    model.train()

    # Data Loader
    train_loader = get_dapt_dataloader(tokenizer)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.dapt_lr, weight_decay=Config.weight_decay
    )

    # Scheduler
    # Adjust total steps for gradient accumulation
    num_update_steps = int(
        len(train_loader) / Config.gradient_accumulation_steps * Config.dapt_epochs
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_update_steps * 0.1),
        num_training_steps=num_update_steps,
    )

    # AMP Scaler
    scaler = torch.amp.GradScaler("cuda")

    # Training Loop
    for epoch in range(Config.dapt_epochs):
        losses = AverageMeter()
        pbar = tqdm(
            train_loader, desc=f"DAPT Epoch {epoch+1}/{Config.dapt_epochs}", leave=False
        )

        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = (
                batch["labels"].to(device)
                if "labels" in batch
                else batch["input_ids"].clone()
            )

            with torch.amp.autocast("cuda"):
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
                # Normalize loss for accumulation
                loss = loss / Config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % Config.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            # Log the raw loss (approximate)
            losses.update(
                loss.item() * Config.gradient_accumulation_steps, input_ids.size(0)
            )
            pbar.set_postfix(loss=losses.avg)

        logger.info(f"DAPT Epoch {epoch+1} - Avg Loss: {losses.avg:.4f}")

    # Save the adapted backbone
    logger.info(f"Saving DAPT model to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Cleanup
    del model, optimizer, scheduler, train_loader, scaler
    torch.cuda.empty_cache()
    gc.collect()


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    """
    Configures Layer-wise Learning Rate Decay (LLRD) for the optimizer.
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = []

    # DeBERTa-v3-large specific layer naming
    # model.model.embeddings...
    # model.model.encoder.layer.0... through .23

    # Define groups
    # 1. Embeddings
    # 2. Encoder Layers (0 to N-1)
    # 3. Task Heads (Pooler, FCs)

    num_layers = 24  # DeBERTa-large has 24 layers
    decay_rate = Config.llrd_decay

    # Initialize groups
    groups = {}
    for i in range(num_layers + 2):  # 0..23 layers + embeddings + head
        groups[i] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Determine layer index
        if "embeddings" in name:
            layer_id = 0  # Bottom
        elif "encoder.layer" in name:
            # Extract layer number
            try:
                # name format example: model.model.encoder.layer.15.output.dense.weight
                parts = name.split(".")
                layer_idx = int(parts[parts.index("layer") + 1])
                layer_id = layer_idx + 1
            except ValueError:
                layer_id = num_layers  # Fallback to top
        else:
            layer_id = num_layers + 1  # Head

        # Determine weight decay
        if any(nd in name for nd in no_decay):
            wd = 0.0
        else:
            wd = weight_decay

        # Calculate LR for this layer
        # Head gets decoder_lr
        # Layers get decayed encoder_lr
        if layer_id == num_layers + 1:
            lr = decoder_lr
        else:
            # Depth from top (0 is top layer in terms of decay calculation usually, but here we go bottom-up)
            # Standard LLRD: lr = base_lr * (decay ** depth)
            # Depth 0 = top encoder layer. Depth 23 = bottom encoder layer.
            # layer_id 1 is bottom encoder (layer 0). layer_id 24 is top encoder (layer 23).
            # We want top encoder to have higher LR than bottom.

            # Distance from top
            distance = num_layers - (
                layer_id - 1
            )  # if layer_id=24 (top), dist=1. if layer_id=1 (bottom), dist=24
            if layer_id == 0:  # Embeddings
                distance = num_layers + 1

            lr = encoder_lr * (decay_rate**distance)

        groups[layer_id].append({"params": p, "weight_decay": wd, "lr": lr})

    # Flatten groups into optimizer list
    for i in range(num_layers + 2):
        if groups[i]:
            optimizer_parameters.extend(
                [
                    {
                        "params": [p["params"]],
                        "weight_decay": p["weight_decay"],
                        "lr": p["lr"],
                    }
                    for p in groups[i]
                ]
            )

    return optimizer_parameters


def train_fn(
    train_loader, model, optimizer, epoch, scheduler, device, loss_fn, awp=None
):
    """
    Executes one training epoch with AMP and Gradient Accumulation.
    """
    model.train()
    losses = AverageMeter()
    scaler = torch.amp.GradScaler("cuda")

    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["labels"].to(device)

        bin_targets = None
        if "bin_labels" in batch:
            bin_targets = batch["bin_labels"].to(device)

        batch_size = input_ids.size(0)

        # Forward with AMP
        with torch.amp.autocast("cuda"):
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, targets, bin_targets)
            loss = loss / Config.gradient_accumulation_steps

        # Backward
        scaler.scale(loss).backward()

        # Adversarial Weight Perturbation (AWP)
        if Config.use_awp and awp is not None and epoch >= Config.awp_start_epoch:
            awp.attack_step()

            with torch.amp.autocast("cuda"):
                adv_outputs = model(input_ids, attention_mask)
                adv_loss = loss_fn(adv_outputs, targets, bin_targets)
                adv_loss = adv_loss / Config.gradient_accumulation_steps

            scaler.scale(adv_loss).backward()
            awp.restore()

        if (step + 1) % Config.gradient_accumulation_steps == 0:
            # Gradient Clipping (must unscale first)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            # Optimizer Step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        losses.update(loss.item() * Config.gradient_accumulation_steps, batch_size)

        if step % Config.print_freq == 0 or step == (len(train_loader) - 1):
            pbar.set_postfix(loss=losses.avg, lr=scheduler.get_last_lr()[0])

    return losses.avg


def valid_fn(val_loader, model, device, loss_fn):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    preds = []
    ground_truth = []

    pbar = tqdm(val_loader, desc="Validation", leave=False)

    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)

            bin_targets = None
            if "bin_labels" in batch:
                bin_targets = batch["bin_labels"].to(device)

            batch_size = input_ids.size(0)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, targets, bin_targets)

            losses.update(loss.item(), batch_size)

            # Collect predictions (Regression logits)
            # Flatten to 1D array
            preds.append(outputs["logits"].view(-1).cpu().numpy())
            ground_truth.append(targets.view(-1).cpu().numpy())

    predictions = np.concatenate(preds)
    true_labels = np.concatenate(ground_truth)

    # Compute Pearson Correlation
    score = get_score(true_labels, predictions)

    return losses.avg, score


def inference_fn(test_loader, model, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds = []

    pbar = tqdm(test_loader, desc="Inference", leave=False)

    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask)

            # Collect regression logits
            preds.append(outputs["logits"].view(-1).cpu().numpy())

    predictions = np.concatenate(preds)

    # Clip predictions to valid range [0, 1]
    predictions = np.clip(predictions, 0, 1)

    return predictions
