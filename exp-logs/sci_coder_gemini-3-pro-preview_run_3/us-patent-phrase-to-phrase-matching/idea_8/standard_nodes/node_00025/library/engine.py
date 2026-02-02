import time
import math
import numpy as np
import torch
import torch.nn as nn
from library.config import Config
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


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    """
    Groups parameters for the optimizer to apply Layer-wise Learning Rate Decay (LLRD).

    Args:
        model (nn.Module): The CustomModel instance.
        encoder_lr (float): Base learning rate for the backbone.
        decoder_lr (float): Learning rate for the custom head.
        weight_decay (float): Weight decay coefficient.

    Returns:
        list: List of parameter groups for the optimizer.
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = []

    # Get number of layers from the backbone config
    # CustomModel wraps the backbone in self.model, but exposes self.config
    num_hidden_layers = model.config.num_hidden_layers

    for name, p in param_optimizer:
        if not p.requires_grad:
            continue

        # Initialize LR with encoder_lr
        lr = encoder_lr

        # 1. Backbone Embeddings (Lowest LR)
        if "embeddings" in name:
            lr = encoder_lr * (Config.llrd_decay**num_hidden_layers)

        # 2. Backbone Encoder Layers (Decaying LR from top to bottom)
        elif "encoder.layer" in name:
            # Example name: model.encoder.layer.11.output.dense.weight
            try:
                # Extract layer index
                # We split by "encoder.layer." and take the number immediately following
                layer_num = int(name.split("encoder.layer.")[1].split(".")[0])

                # Layer 0 is bottom, Layer N-1 is top.
                # We want top layers to have higher LR (closer to encoder_lr)
                # Formula: lr = base * decay ^ (max_layers - 1 - current_layer)
                lr = encoder_lr * (
                    Config.llrd_decay ** (num_hidden_layers - 1 - layer_num)
                )
            except Exception:
                # Fallback if naming convention differs
                lr = encoder_lr

        # 3. Custom Head (Pooler, FCs) (Highest LR)
        # These parameters are initialized randomly or specifically for the task
        elif any(n in name for n in ["pooler", "attention_pooler", "fc", "fc_class"]):
            lr = decoder_lr

        # Apply Weight Decay
        if any(nd in name for nd in no_decay):
            wd = 0.0
        else:
            wd = weight_decay

        optimizer_parameters.append({"params": [p], "weight_decay": wd, "lr": lr})

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
    scaler=None,
):
    """
    Performs one epoch of training.

    Args:
        fold (int): Current fold number.
        train_loader (DataLoader): Training data loader.
        model (nn.Module): The model to train.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        epoch (int): Current epoch number.
        scheduler (Scheduler): Learning rate scheduler.
        device (torch.device): Device to train on.
        awp (AWP, optional): Adversarial Weight Perturbation object.
        scaler (GradScaler, optional): Gradient scaler for mixed precision.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()
    start = time.time()

    for step, batch in enumerate(train_loader):
        # Move inputs to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        batch_size = labels.size(0)

        # Forward Pass with Mixed Precision (if scaler provided)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(input_ids, attention_mask)
            loss_dict = criterion(outputs, labels)
            loss = loss_dict["loss"]

            if Config.gradient_accumulation_steps > 1:
                loss = loss / Config.gradient_accumulation_steps

        # Backward Pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Adversarial Weight Perturbation (AWP)
        # Only apply if AWP object exists and we are past the start epoch
        if awp is not None and epoch >= Config.awp_start_epoch:
            awp.attack(epoch)

            # Second forward pass with perturbed weights
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs_adv = model(input_ids, attention_mask)
                loss_dict_adv = criterion(outputs_adv, labels)
                loss_adv = loss_dict_adv["loss"]
                if Config.gradient_accumulation_steps > 1:
                    loss_adv = loss_adv / Config.gradient_accumulation_steps

            # Second backward pass
            if scaler is not None:
                scaler.scale(loss_adv).backward()
            else:
                loss_adv.backward()

            # Restore original weights
            awp.restore()

        # Optimizer Step
        if (step + 1) % Config.gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
                optimizer.step()

            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        losses.update(loss_dict["loss"].item(), batch_size)

        if step % Config.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                f"Epoch: [{epoch + 1}][{step}/{len(train_loader)}] "
                f"Elapsed: {time.time() - start:.1f}s "
                f"Loss: {losses.val:.4f} "
                f"Avg Loss: {losses.avg:.4f} "
                f"LR: {optimizer.param_groups[0]['lr']:.8f}"
            )

    return losses.avg


def valid_fn(val_loader, model, criterion, device):
    """
    Performs validation on the validation set.

    Args:
        val_loader (DataLoader): Validation data loader.
        model (nn.Module): The model to evaluate.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to evaluate on.

    Returns:
        tuple: (average_loss, pearson_score)
    """
    model.eval()
    losses = AverageMeter()
    preds = []
    targets = []
    start = time.time()

    for step, batch in enumerate(val_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        batch_size = labels.size(0)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            loss_dict = criterion(outputs, labels)
            loss = loss_dict["loss"]

        losses.update(loss.item(), batch_size)

        # Store predictions (logits) and targets for metric calculation
        # outputs['logits'] is (B, 1), squeeze to (B,)
        preds.append(outputs["logits"].squeeze(-1).to("cpu").numpy())
        targets.append(labels.to("cpu").numpy())

    predictions = np.concatenate(preds)
    ground_truth = np.concatenate(targets)

    # Calculate Pearson Correlation
    score = get_score(ground_truth, predictions)

    print(
        f"EVAL: [{len(val_loader)}] "
        f"Elapsed: {time.time() - start:.1f}s "
        f"Loss: {losses.avg} "
        f"Score: {score}"
    )

    return losses.avg, score


def inference_fn(test_loader, model, device):
    """
    Generates predictions for the test set.

    Args:
        test_loader (DataLoader): Test data loader.
        model (nn.Module): The model to use for inference.
        device (torch.device): Device to run on.

    Returns:
        np.array: Array of predicted scores.
    """
    model.eval()
    preds = []

    for step, batch in enumerate(test_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)

        preds.append(outputs["logits"].squeeze(-1).to("cpu").numpy())

    predictions = np.concatenate(preds)

    # Clip predictions to [0, 1] range as per task requirements
    predictions = np.clip(predictions, 0, 1)

    return predictions
