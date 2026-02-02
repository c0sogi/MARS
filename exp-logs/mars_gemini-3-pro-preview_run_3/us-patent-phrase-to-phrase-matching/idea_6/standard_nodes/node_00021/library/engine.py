import torch
import torch.nn as nn
import numpy as np
import time
from library.config import cfg
from library.utils import AverageMeter, get_score, get_logger

# Initialize logger for this module
logger = get_logger("engine.log")


class AWP:
    """
    Adversarial Weight Perturbation (AWP).
    Perturbs model weights to maximize loss, improving generalization.
    """

    def __init__(self, model, optimizer, adv_param="weight", adv_lr=1.0, adv_eps=0.01):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, inputs, labels, attention_mask, criterion, epoch):
        """
        Performs the AWP attack and backward pass.
        Note: This method is a helper; in the main loop we control the flow explicitly
        to handle the custom batch structure and composite loss.
        """
        pass

    def _save(self):
        """Save original weights."""
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    self.backup_eps[name] = param.data.clone()

    def _restore(self):
        """Restore original weights."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

    def attack(self):
        """Apply perturbation to weights based on gradients."""
        self._save()
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
                    # Calculate perturbation: direction * step_size * scale
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    # Clamp perturbation magnitude to epsilon
                    # Note: We simplify here by trusting adv_lr/eps config or implementing clipping if needed.
                    # Standard AWP often just adds the scaled gradient.
                    param.data.add_(r_at)

                    # Optional: Projection to epsilon ball could be added here if strict constraint is needed
                    # but adv_lr usually controls the step size sufficiently for AWP.

    def restore(self):
        """Restore weights to original state."""
        self._restore()


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    """
    Configures layer-wise learning rate decay (LLRD) for the optimizer.
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = []

    # 1. Separate Backbone and Head
    # DeBERTa-v3 structure: backbone.embeddings, backbone.encoder.layer.0 ... .layer.23

    # Get number of layers from config if available, else infer
    if hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers
    else:
        # Fallback for DeBERTa large usually 24
        num_layers = 24

    # Define Layer Groups
    # Group 0: Embeddings (Lowest LR)
    # Group 1..N: Encoder Layers (Increasing LR)
    # Group N+1: Head / Custom Layers (Highest LR = decoder_lr)

    # Initialize groups
    # We map layer index to a list of params
    # Index -1 for embeddings, 0 to num_layers-1 for encoder layers, num_layers for head
    grouped_params = {i: [] for i in range(-1, num_layers + 1)}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if "backbone" not in name:
            # Head parameters
            grouped_params[num_layers].append((name, p))
        elif "embeddings" in name:
            # Embeddings
            grouped_params[-1].append((name, p))
        elif "encoder.layer" in name:
            # Encoder Layers
            # name format: backbone.encoder.layer.15.output...
            try:
                layer_idx = int(name.split("encoder.layer.")[1].split(".")[0])
                grouped_params[layer_idx].append((name, p))
            except:
                # Fallback to head or specific handling
                grouped_params[num_layers].append((name, p))
        else:
            # Other backbone params (e.g. final layer norm, rel_embeddings)
            # Treat as top layer of backbone
            grouped_params[num_layers - 1].append((name, p))

    # Create Optimizer List
    for layer_idx in range(-1, num_layers + 1):
        # Calculate LR for this layer
        if layer_idx == num_layers:
            lr = decoder_lr
        elif layer_idx == -1:
            lr = encoder_lr * (cfg.llrd_decay ** (num_layers + 1))
        else:
            # Layer 0 is furthest from head, Layer 23 is closest
            # Decay relative to head
            distance_from_head = num_layers - 1 - layer_idx
            lr = encoder_lr * (cfg.llrd_decay ** (distance_from_head + 1))

        # Split into decay and no_decay
        params_decay = []
        params_no_decay = []

        for name, p in grouped_params[layer_idx]:
            if any(nd in name for nd in no_decay):
                params_no_decay.append(p)
            else:
                params_decay.append(p)

        if params_decay:
            optimizer_parameters.append(
                {"params": params_decay, "lr": lr, "weight_decay": weight_decay}
            )
        if params_no_decay:
            optimizer_parameters.append(
                {"params": params_no_decay, "lr": lr, "weight_decay": 0.0}
            )

    return optimizer_parameters


def train_fn(
    fold, train_loader, model, criterion, optimizer, epoch, scheduler, device, awp=None
):
    """
    Executes one training epoch.
    """
    model.train()
    scaler = torch.amp.GradScaler("cuda")

    losses = AverageMeter()
    scores = AverageMeter()

    start_time = time.time()

    for step, batch in enumerate(train_loader):
        # Move batch to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        batch_size = batch["input_ids"].size(0)

        # --- Forward Pass ---
        with torch.amp.autocast("cuda"):
            outputs = model(batch["input_ids"], batch["attention_mask"])
            loss_dict = criterion(outputs, batch)
            loss = loss_dict["loss"]

            # Normalize loss for gradient accumulation
            if cfg.gradient_accumulation_steps > 1:
                loss = loss / cfg.gradient_accumulation_steps

        # --- Backward Pass ---
        scaler.scale(loss).backward()

        # --- AWP Attack ---
        if awp is not None and epoch >= cfg.awp_start_epoch:
            # 1. Save weights and perturb
            awp.attack()

            # 2. Forward pass with perturbed weights
            with torch.amp.autocast("cuda"):
                outputs_awp = model(batch["input_ids"], batch["attention_mask"])
                loss_dict_awp = criterion(outputs_awp, batch)
                loss_awp = loss_dict_awp["loss"]

                if cfg.gradient_accumulation_steps > 1:
                    loss_awp = loss_awp / cfg.gradient_accumulation_steps

            # 3. Backward pass with perturbed weights (accumulate gradients)
            scaler.scale(loss_awp).backward()

            # 4. Restore original weights
            awp.restore()

        # --- Optimizer Step ---
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None and cfg.batch_scheduler:
                scheduler.step()

        # --- Metrics ---
        losses.update(loss_dict["loss"].item(), batch_size)

        # Calculate batch score for monitoring
        preds = outputs["logits"].detach().cpu().numpy().flatten()
        labels = batch["labels"].detach().cpu().numpy().flatten()

        # Handle case where batch size is small or constant values (pearson undefined)
        try:
            if len(np.unique(preds)) > 1 and len(np.unique(labels)) > 1:
                batch_score = get_score(labels, preds)
                scores.update(batch_score, batch_size)
        except:
            pass

        if step % cfg.print_freq == 0 or step == (len(train_loader) - 1):
            logger.info(
                f"Epoch: [{epoch + 1}][{step}/{len(train_loader)}] "
                f"Loss: {losses.val:.4f}({losses.avg:.4f}) "
                f"Score: {scores.val:.4f}({scores.avg:.4f}) "
                f"LR: {optimizer.param_groups[0]['lr']:.8f}"
            )

    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    preds_list = []
    labels_list = []

    start_time = time.time()

    with torch.no_grad():
        for step, batch in enumerate(valid_loader):
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            batch_size = batch["input_ids"].size(0)

            # Forward
            with torch.amp.autocast("cuda"):
                outputs = model(batch["input_ids"], batch["attention_mask"])
                loss_dict = criterion(outputs, batch)

            # Record Loss
            losses.update(loss_dict["loss"].item(), batch_size)

            # Collect predictions
            preds = outputs["logits"].view(-1).cpu().numpy()
            labels = batch["labels"].view(-1).cpu().numpy()

            preds_list.append(preds)
            labels_list.append(labels)

    # Concatenate all predictions
    predictions = np.concatenate(preds_list)
    ground_truth = np.concatenate(labels_list)

    # Compute Global Pearson Score
    score = get_score(ground_truth, predictions)

    return losses.avg, score, predictions
