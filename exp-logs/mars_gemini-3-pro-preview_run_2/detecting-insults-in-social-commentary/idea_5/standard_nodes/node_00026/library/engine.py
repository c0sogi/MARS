import torch
import torch.nn as nn
import numpy as np
import time
from library.utils import AverageMeter, get_score
from library.config import Config


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    """
    Configures the optimizer parameters with Layer-wise Learning Rate Decay (LLRD).
    Groups parameters into Embeddings, Encoder Layers (0-23), and Head (Decoder).
    """
    # Define parameter groups
    optimizer_parameters = []

    # Parameters to exclude from weight decay
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # Get number of layers from config if possible, else default to 24 (Large)
    num_layers = 24
    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers
    elif hasattr(model, "model") and hasattr(model.model.config, "num_hidden_layers"):
        num_layers = model.model.config.num_hidden_layers

    # Pre-calculate Learning Rates for each layer
    # layer_lrs[i] = encoder_lr * (decay ** (num_layers - 1 - i))
    # embeddings_lr = encoder_lr * (decay ** num_layers)
    layer_lrs = {}
    for i in range(num_layers):
        layer_lrs[i] = encoder_lr * (Config.llrd_decay ** (num_layers - 1 - i))

    embeddings_lr = encoder_lr * (Config.llrd_decay**num_layers)

    # Iterate through all parameters
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Determine Learning Rate based on parameter name
        if "model" not in name:
            # Head parameters (fc, pool, etc.)
            cur_lr = decoder_lr
        elif "embeddings" in name:
            # Embedding layer
            cur_lr = embeddings_lr
        elif "encoder.layer" in name:
            # Encoder layers
            try:
                # Format: model.encoder.layer.15.output...
                parts = name.split(".")
                # Find the index after 'layer'
                layer_idx = int(parts[parts.index("layer") + 1])
                cur_lr = layer_lrs.get(layer_idx, encoder_lr)
            except (ValueError, IndexError):
                # Fallback if parsing fails
                cur_lr = encoder_lr
        else:
            # Other backbone parameters
            cur_lr = encoder_lr

        # Determine Weight Decay
        if any(nd in name for nd in no_decay):
            cur_wd = 0.0
        else:
            cur_wd = weight_decay

        # Add to existing group or create new one
        found = False
        for group in optimizer_parameters:
            if group["lr"] == cur_lr and group["weight_decay"] == cur_wd:
                group["params"].append(p)
                found = True
                break

        if not found:
            optimizer_parameters.append(
                {"params": [p], "lr": cur_lr, "weight_decay": cur_wd}
            )

    return optimizer_parameters


def train_fn(
    train_loader, model, criterion, optimizer, epoch, scheduler, device, awp=None
):
    """
    Training loop with Mixed Precision, Gradient Accumulation, and AWP.
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    losses = AverageMeter()

    # Global step for logging/tracking
    global_step = 0

    for step, inputs in enumerate(train_loader):
        # Move inputs to device
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        labels = inputs["target"]
        # Remove target from inputs for forward pass
        forward_inputs = {k: v for k, v in inputs.items() if k != "target"}

        batch_size = labels.size(0)

        # Mixed Precision Forward
        with torch.cuda.amp.autocast():
            y_preds = model(**forward_inputs)
            loss = criterion(y_preds.view(-1), labels)

            # Scale Loss for Gradient Accumulation
            if Config.gradient_accumulation_steps > 1:
                loss = loss / Config.gradient_accumulation_steps

        # Backward Pass (Clean)
        scaler.scale(loss).backward()

        # Adversarial Weight Perturbation (AWP)
        if awp is not None:
            # Perturb weights based on current gradients
            awp.attack()

            with torch.cuda.amp.autocast():
                # Forward pass with perturbed weights
                y_preds_adv = model(**forward_inputs)
                loss_adv = criterion(y_preds_adv.view(-1), labels)

                if Config.gradient_accumulation_steps > 1:
                    loss_adv = loss_adv / Config.gradient_accumulation_steps

            # Backward pass (Adversarial) - accumulates gradients
            scaler.scale(loss_adv).backward()

            # Restore original weights
            awp.restore()

        # Optimizer Step (with Gradient Accumulation)
        if (step + 1) % Config.gradient_accumulation_steps == 0:
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            # Update weights
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

            global_step += 1

        # Update metrics (restore original loss scale for logging)
        losses.update(loss.item() * Config.gradient_accumulation_steps, batch_size)

        if step % Config.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                f"Epoch: [{epoch + 1}][{step}/{len(train_loader)}] "
                f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                f"LR: {optimizer.param_groups[0]['lr']:.8f}"
            )

    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    """
    Validation loop. Calculates Loss and AUC.
    """
    model.eval()
    preds = []
    targets = []
    losses = AverageMeter()

    for step, inputs in enumerate(valid_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        labels = inputs["target"]
        forward_inputs = {k: v for k, v in inputs.items() if k != "target"}
        batch_size = labels.size(0)

        with torch.no_grad():
            y_preds = model(**forward_inputs)
            loss = criterion(y_preds.view(-1), labels)

        if Config.gradient_accumulation_steps > 1:
            loss = loss / Config.gradient_accumulation_steps

        losses.update(loss.item() * Config.gradient_accumulation_steps, batch_size)

        # Apply sigmoid to get probabilities
        preds.append(y_preds.sigmoid().to("cpu").numpy())
        targets.append(labels.to("cpu").numpy())

    predictions = np.concatenate(preds)
    targets = np.concatenate(targets)

    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Calculate AUC
    score = get_score(targets, predictions)

    # Print full precision as requested
    print(f"EVAL: Loss: {losses.avg} AUC: {score}")

    return score, predictions, losses.avg
