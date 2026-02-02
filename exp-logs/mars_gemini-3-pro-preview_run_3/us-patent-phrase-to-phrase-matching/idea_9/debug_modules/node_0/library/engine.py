import torch
import torch.nn as nn
import numpy as np
import time
from library.config import Config
from library.loss import HybridLoss
from library.utils import get_logger

logger = get_logger()


class AWP:
    """
    Adversarial Weight Perturbation (AWP).
    Perturbs model weights to maximize the loss, encouraging the model to find
    flatter minima in the loss landscape.
    """

    def __init__(self, model, optimizer, adv_lr, adv_eps):
        self.model = model
        self.optimizer = optimizer
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack(self):
        """
        Performs the adversarial attack on the model weights.
        Saves current weights and applies perturbation based on gradients.
        """
        e = 1e-6
        self._save()
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_lr != 0:
                grad_norm = torch.norm(param.grad)
                if grad_norm > 0 and not torch.isnan(grad_norm):
                    weight_norm = torch.norm(param.data)

                    # Compute perturbation: eta * (grad / |grad|) * |weight|
                    perturbation = (
                        self.adv_lr * param.grad / (grad_norm + e) * (weight_norm + e)
                    )

                    # Apply perturbation
                    param.data.add_(perturbation)

                    # Projection/Clipping to ensure weights stay within epsilon ball
                    min_val = self.backup_eps[name][0]
                    max_val = self.backup_eps[name][1]
                    param.data = torch.max(torch.min(param.data, max_val), min_val)

    def _save(self):
        """
        Saves the current model weights and computes the clipping bounds.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.data.abs()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def restore(self):
        """
        Restores the original model weights.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


class EMA:
    """
    Exponential Moving Average (EMA) of model parameters.
    Maintains a shadow copy of weights that are updated via a moving average.
    """

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """
        Registers the initial model parameters to track.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """
        Updates the shadow weights using the current model weights.
        Formula: shadow = decay * shadow + (1 - decay) * current
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """
        Replaces the model's current weights with the shadow (EMA) weights.
        Used during validation or inference.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """
        Restores the original weights (saved in apply_shadow).
        Used to resume training after validation.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def train_fn(dataloader, model, optimizer, scheduler, device, epoch, ema=None):
    """
    Training loop for a single epoch.

    Args:
        dataloader: PyTorch DataLoader for training data.
        model: The model to train.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        device: Device to run training on (cuda/cpu).
        epoch: Current epoch number.
        ema: Instance of EMA class (optional).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    criterion = HybridLoss()
    scaler = torch.cuda.amp.GradScaler()

    # Initialize AWP
    awp = None
    if Config.use_awp and epoch >= Config.awp_start_epoch:
        awp = AWP(model, optimizer, adv_lr=Config.awp_lr, adv_eps=Config.awp_eps)

    running_loss = 0.0
    dataset_size = 0

    start_time = time.time()

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["labels"].to(device)

        batch_size = input_ids.size(0)

        # 1. Forward Pass & Loss Calculation
        with torch.cuda.amp.autocast():
            outputs = model(input_ids, attention_mask)
            loss, metrics = criterion(outputs, targets)

        if Config.gradient_accumulation_steps > 1:
            loss = loss / Config.gradient_accumulation_steps

        # 2. Backward Pass (Compute gradients for clean weights)
        scaler.scale(loss).backward()

        if (step + 1) % Config.gradient_accumulation_steps == 0:

            # 3. AWP Attack (Adversarial Training)
            if awp is not None:
                # Unscale gradients before using them for perturbation
                scaler.unscale_(optimizer)

                # Perturb weights
                awp.attack()

                # Forward pass with perturbed weights
                with torch.cuda.amp.autocast():
                    outputs_adv = model(input_ids, attention_mask)
                    loss_adv, _ = criterion(outputs_adv, targets)

                # We want to update weights to minimize the adversarial loss.
                # Clear previous gradients (from clean weights) to avoid double counting
                # or mixing directions excessively.
                model.zero_grad()

                # Backward pass with perturbed weights
                scaler.scale(loss_adv).backward()

                # Restore original weights (but keep the gradients computed from adv loss)
                awp.restore()

            # 4. Optimizer Step
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

            # 5. EMA Update
            if ema is not None:
                ema.update()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        if (step + 1) % Config.print_freq == 0 or (step + 1) == len(dataloader):
            logger.info(
                f"Epoch {epoch+1} | Step {step+1}/{len(dataloader)} | "
                f"Loss: {running_loss / dataset_size:.6f}"
            )

    return running_loss / dataset_size


def valid_fn(dataloader, model, device, ema=None):
    """
    Validation loop.

    Args:
        dataloader: PyTorch DataLoader for validation data.
        model: The model to evaluate.
        device: Device to run evaluation on.
        ema: Instance of EMA class. If provided, uses EMA weights for inference.

    Returns:
        tuple: (average_loss, pearson_score, predictions)
    """
    # Apply EMA weights if available
    if ema:
        ema.apply_shadow()

    model.eval()
    criterion = HybridLoss()

    running_loss = 0.0
    dataset_size = 0

    preds = []
    labels_list = []

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            batch_size = input_ids.size(0)

            outputs = model(input_ids, attention_mask)

            # Store predictions
            # outputs['logits'] is the regression score
            batch_preds = outputs["logits"].view(-1).cpu().numpy()
            preds.append(batch_preds)

            # If labels exist (Validation), calculate loss and store labels
            if "labels" in batch:
                targets = batch["labels"].to(device)
                loss, _ = criterion(outputs, targets)
                running_loss += loss.item() * batch_size
                labels_list.append(targets.view(-1).cpu().numpy())

            dataset_size += batch_size

    # Restore original weights
    if ema:
        ema.restore()

    all_preds = np.concatenate(preds)

    # Calculate metrics if labels were present
    if len(labels_list) > 0:
        all_labels = np.concatenate(labels_list)
        avg_loss = running_loss / dataset_size

        # Pearson Correlation
        pearson_score = np.corrcoef(all_preds, all_labels)[0, 1]
    else:
        avg_loss = 0.0
        pearson_score = 0.0

    return avg_loss, pearson_score, all_preds
