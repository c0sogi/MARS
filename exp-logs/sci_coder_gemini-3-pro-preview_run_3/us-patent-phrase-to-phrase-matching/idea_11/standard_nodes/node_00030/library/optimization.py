import torch
import torch.nn as nn
from library.config import CFG
from library.utils import get_logger

logger = get_logger("optimization.log")


def get_optimizer_params(model, encoder_lr, head_lr, weight_decay):
    """
    Configures layer-wise learning rate decay (LLRD) and weight decay groups.

    Args:
        model (nn.Module): The model to optimize.
        encoder_lr (float): Base learning rate for the encoder's top layer.
        head_lr (float): Learning rate for the custom heads.
        weight_decay (float): Weight decay coefficient.

    Returns:
        list: List of parameter groups for the optimizer.
    """
    # Exclude bias and LayerNorm from weight decay
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = []

    # 1. Handle Backbone (DeBERTa) with LLRD
    if hasattr(model, "backbone"):
        backbone = model.backbone

        # DeBERTa structure: embeddings -> encoder.layer (ModuleList)
        # We want to decay from Top (Output) -> Bottom (Input)

        # Get layers: [Embeddings, Layer 0, Layer 1, ..., Layer N]
        layers = [backbone.embeddings] + list(backbone.encoder.layer)

        # Reverse to process Top -> Bottom
        layers.reverse()

        lr = encoder_lr

        for layer in layers:
            # Separate decay and no_decay params within this layer
            decay_params = []
            no_decay_params = []

            for name, param in layer.named_parameters():
                if not param.requires_grad:
                    continue

                if any(nd in name for nd in no_decay):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

            # Add groups
            if decay_params:
                optimizer_parameters.append(
                    {"params": decay_params, "weight_decay": weight_decay, "lr": lr}
                )
            if no_decay_params:
                optimizer_parameters.append(
                    {"params": no_decay_params, "weight_decay": 0.0, "lr": lr}
                )

            # Decay LR for the next layer (going deeper)
            lr *= CFG.llrd_decay

    # 2. Handle Custom Heads (Mixing, Pooler, FCs)
    # Identify backbone parameters to exclude them from this pass
    backbone_param_ids = set(id(p) for p in model.backbone.parameters())

    head_params_decay = []
    head_params_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Skip if part of backbone
        if id(param) in backbone_param_ids:
            continue

        if any(nd in name for nd in no_decay):
            head_params_no_decay.append(param)
        else:
            head_params_decay.append(param)

    if head_params_decay:
        optimizer_parameters.append(
            {"params": head_params_decay, "weight_decay": weight_decay, "lr": head_lr}
        )

    if head_params_no_decay:
        optimizer_parameters.append(
            {"params": head_params_no_decay, "weight_decay": 0.0, "lr": head_lr}
        )

    return optimizer_parameters


class AWP:
    """
    Adversarial Weight Perturbation (AWP).
    Perturbs weights in the direction of the gradient ascent to flatten the loss landscape.
    """

    def __init__(self, model, optimizer, adv_param="weight", adv_lr=None, adv_eps=None):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr if adv_lr is not None else CFG.awp_lr
        self.adv_eps = adv_eps if adv_eps is not None else CFG.awp_eps
        self.backup = {}
        self.backup_eps = {}

    def _save(self):
        """Save original weights and initialize epsilon backup."""
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

    def attack_step(self):
        """
        Perturbs the model weights based on gradients.
        Should be called after loss.backward() so gradients are available.
        """
        e = 1e-6
        self._save()

        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                # Calculate gradient norm and data norm
                grad_norm = torch.norm(param.grad)
                data_norm = torch.norm(param.data)

                if grad_norm != 0 and not torch.isnan(grad_norm):
                    # Calculate perturbation: scale gradient by data magnitude
                    # delta = lr * (grad / |grad|) * |data|
                    r_at = self.adv_lr * param.grad / (grad_norm + e) * (data_norm + e)

                    # Add perturbation
                    param.data.add_(r_at)

                    # Clamp perturbation within epsilon ball around the original weights
                    # min(max(data, orig - eps), orig + eps)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name] - self.adv_eps),
                        self.backup_eps[name] + self.adv_eps,
                    )

    def restore(self):
        """Restores the original model weights."""
        self._restore()


class EMA:
    """
    Exponential Moving Average of model weights.
    Maintains a shadow copy of the model parameters for robust inference.
    """

    def __init__(self, model, decay=None):
        self.model = model
        self.decay = decay if decay is not None else CFG.ema_decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """Initialize EMA weights (shadow copy)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """
        Update EMA weights based on current model weights.
        theta_ema = decay * theta_ema + (1 - decay) * theta_curr
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """
        Replace model weights with EMA weights (for validation/inference).
        Backs up original weights first.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """
        Restore original model weights (for continuing training).
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}
