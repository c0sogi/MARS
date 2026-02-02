import torch
import torch.nn as nn
from collections import defaultdict
from library.config import Config


class AWP:
    """
    Adversarial Weight Perturbation (AWP) implementation.
    Perturbs model weights in the direction of the gradient to flatten the loss landscape.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        adv_lr: float,
        adv_eps: float,
        start_epoch: float,
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.backup = {}
        self.backup_eps = {}

    def attack(self):
        """
        Performs the adversarial attack on the model weights.
        Saves the original weights and applies the perturbation.
        """
        self._save()
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_lr != 0:
                # Get gradient and compute norm
                grad = param.grad
                norm = torch.norm(grad)

                # Check for validity
                if norm != 0 and not torch.isnan(norm):
                    # Compute perturbation: r = eta * g / (||g|| + eps)
                    # We add perturbation to move in direction of gradient (ascent)
                    # to maximize loss, then the optimizer minimizes this harder loss.
                    r_at = self.adv_lr * grad / (norm + self.adv_eps)
                    param.data.add_(r_at)

    def restore(self):
        """
        Restores the original model weights from backup.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

    def _save(self):
        """
        Saves the current model weights.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_lr != 0:
                self.backup[name] = param.data.clone()


class EMA:
    """
    Exponential Moving Average (EMA) for model parameters.
    Maintains a shadow copy of weights that is updated smoothly.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """
        Initialize shadow weights with current model weights.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """
        Update shadow weights: shadow = decay * shadow + (1 - decay) * param
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """
        Replace model weights with shadow weights (for validation/inference).
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


def get_optimizer_params(model: nn.Module, cfg: Config):
    """
    Constructs the parameter groups for the optimizer with Layer-wise Learning Rate Decay (LLRD).

    Args:
        model: The CustomModel instance.
        cfg: Configuration object containing learning rates and decay settings.

    Returns:
        List of dictionaries containing parameter groups.
    """
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

    # Base configuration
    encoder_lr = cfg.encoder_lr
    head_lr = cfg.head_lr
    weight_decay = cfg.weight_decay
    llrd_rate = cfg.llrd_rate

    # Get number of layers from backbone config
    # model.model is the AutoModel (DeBERTa)
    num_hidden_layers = model.config.num_hidden_layers

    # Group parameters by (learning_rate, weight_decay)
    groups = defaultdict(list)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 1. Determine Learning Rate
        if name.startswith("model."):
            # This belongs to the backbone (DeBERTa)
            if "embeddings" in name:
                # Embeddings are at the very bottom
                depth = num_hidden_layers
            elif "encoder.layer" in name:
                # Extract layer index
                # Format: model.encoder.layer.X. ...
                parts = name.split(".")
                layer_idx = 0
                for i, part in enumerate(parts):
                    if part == "layer":
                        try:
                            layer_idx = int(parts[i + 1])
                        except (ValueError, IndexError):
                            layer_idx = 0
                        break

                # Calculate depth from top (0) to bottom (num_layers - 1)
                # Layer 23 (top) -> depth 0
                # Layer 0 (bottom) -> depth 23
                depth = num_hidden_layers - 1 - layer_idx
            else:
                # Other backbone parameters (e.g. final layer norm, pooler if part of backbone)
                # Treat as top layer
                depth = 0

            # Apply LLRD
            cur_lr = encoder_lr * (llrd_rate**depth)

        else:
            # Custom Head parameters (pooler, heads, etc.)
            cur_lr = head_lr

        # 2. Determine Weight Decay
        if any(nd in name for nd in no_decay):
            cur_wd = 0.0
        else:
            cur_wd = weight_decay

        # Add to group
        groups[(cur_lr, cur_wd)].append(param)

    # Convert groups to list format required by optimizer
    optimizer_params = []
    for (lr, wd), params in groups.items():
        optimizer_params.append({"params": params, "lr": lr, "weight_decay": wd})

    return optimizer_params
