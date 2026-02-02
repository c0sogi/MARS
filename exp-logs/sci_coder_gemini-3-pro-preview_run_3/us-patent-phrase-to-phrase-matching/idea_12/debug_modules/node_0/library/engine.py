import torch
import time
import math
import numpy as np
import torch.nn as nn
from library.utils import AverageMeter, get_score, get_logger
from library.config import CFG


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    """
    Configures optimizer parameters with Layer-wise Learning Rate Decay (LLRD).
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = []

    # Access the HF backbone configuration to get the number of layers
    # model.model is the AutoModel (DebertaV3)
    try:
        num_layers = model.model.config.num_hidden_layers
    except AttributeError:
        # Fallback if structure is different, though model.py defines it as self.model
        num_layers = 24

    layer_decay = CFG.layer_decay

    # Initialize groups
    # 0: Embeddings (lowest LR)
    # 1..N: Encoder Layers (increasing LR)
    # N+1: Head / Custom Layers (decoder_lr)

    groups = {}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Determine the layer ID for LLRD
        if "model.embeddings" in name:
            layer_id = 0
        elif "model.encoder.layer" in name:
            # Extract layer index from name, e.g., "model.encoder.layer.11.output..."
            try:
                layer_idx = int(name.split("model.encoder.layer.")[1].split(".")[0])
                layer_id = layer_idx + 1
            except:
                layer_id = 0
        else:
            # Custom layers (pooler, fc_score, layer_pooling, etc.)
            layer_id = num_layers + 1

        # Determine Learning Rate for this group
        if layer_id <= num_layers:
            # Backbone layers
            lr = encoder_lr * (layer_decay ** (num_layers - layer_id))
        else:
            # Head layers
            lr = decoder_lr

        # Determine Weight Decay
        if any(nd in name for nd in no_decay):
            wd = 0.0
        else:
            wd = weight_decay

        # Group key: (lr, weight_decay)
        group_key = (lr, wd)
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(p)

    # Create final parameter list
    for (lr, wd), params in groups.items():
        optimizer_parameters.append({"params": params, "weight_decay": wd, "lr": lr})

    return optimizer_parameters


def train_fn(
    fold,
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    scheduler,
    device,
    awp=None,
    ema=None,
    config=CFG,
):
    """
    Training loop for one epoch.
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)
    losses = AverageMeter()
    start = time.time()

    # Logger
    logger = get_logger(f"{config.working_dir}/train.log")

    for step, (inputs, labels) in enumerate(train_loader):
        # Move inputs to device
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        # Forward Pass with Mixed Precision
        with torch.cuda.amp.autocast(enabled=config.fp16):
            y_preds = model(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs.get("token_type_ids"),
            )
            loss = criterion(y_preds, labels)

        # Record Loss
        if config.gradient_accumulation_steps > 1:
            loss = loss / config.gradient_accumulation_steps

        losses.update(loss.item(), batch_size)

        # Backward Pass
        scaler.scale(loss).backward()

        if (step + 1) % config.gradient_accumulation_steps == 0:

            # Adversarial Weight Perturbation (AWP)
            if awp is not None and epoch >= config.awp_start_epoch:
                # Save weights, inject noise
                awp.attack_step()

                # Forward pass with perturbed weights
                with torch.cuda.amp.autocast(enabled=config.fp16):
                    y_preds_adv = model(
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        inputs.get("token_type_ids"),
                    )
                    loss_adv = criterion(y_preds_adv, labels)

                # Backward pass for adversarial loss
                scaler.scale(loss_adv).backward()

                # Restore original weights
                awp.restore()

            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            # Optimizer Step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Scheduler Step
            if scheduler is not None:
                scheduler.step()

            # EMA Update
            if ema is not None:
                ema.update()

        # Logging
        if step % config.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                f"Epoch: [{epoch + 1}][{step}/{len(train_loader)}] "
                f"Elapsed: {time.time() - start:.1f}s "
                f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

    return losses.avg


def valid_fn(valid_loader, model, criterion, device, config=CFG):
    """
    Validation loop.
    """
    model.eval()
    losses = AverageMeter()
    preds = []
    targets = []
    start = time.time()

    for step, (inputs, labels) in enumerate(valid_loader):
        # Move inputs to device
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        # Forward Pass (No Grad)
        with torch.no_grad():
            y_preds = model(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs.get("token_type_ids"),
            )
            loss = criterion(y_preds, labels)

        # Record Loss
        if config.gradient_accumulation_steps > 1:
            loss = loss / config.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)

        # Collect predictions for metric calculation
        # y_preds is a dict containing 'score' and 'logits'
        score_pred = y_preds["score"].view(-1).to("cpu").numpy()
        preds.append(score_pred)
        targets.append(labels.view(-1).to("cpu").numpy())

        if step % config.print_freq == 0 or step == (len(valid_loader) - 1):
            print(
                f"EVAL: [{step}/{len(valid_loader)}] "
                f"Elapsed: {time.time() - start:.1f}s "
                f"Loss: {losses.val:.4f} ({losses.avg:.4f})"
            )

    predictions = np.concatenate(preds)
    labels = np.concatenate(targets)

    # Calculate Pearson Correlation
    score = get_score(labels, predictions)

    return losses.avg, score
