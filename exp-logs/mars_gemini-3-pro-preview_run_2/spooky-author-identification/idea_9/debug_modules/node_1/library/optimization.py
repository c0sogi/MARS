import torch
import torch.nn as nn
from collections import defaultdict
from library.config import Config


class AWP:
    """
    Adversarial Weight Perturbation (AWP).

    This class implements the AWP technique which injects adversarial noise
    into the model weights to flatten the loss landscape and improve generalization.
    It is typically used in the training loop as follows:
    1. Forward + Backward (Standard)
    2. awp.attack() -> perturb weights based on gradients
    3. Forward + Backward (Adversarial)
    4. awp.restore() -> restore original weights
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=Config.awp_lr,
        adv_eps=Config.awp_eps,
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack_step(self):
        """
        Performs the adversarial attack on the model weights.
        Saves original weights and applies perturbation.
        """
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                # Save original parameters
                self.backup[name] = param.data.clone()

                # Calculate perturbation
                grad_norm = torch.norm(param.grad)
                if grad_norm > 0:
                    # AWP formula: delta = eps * weight * grad / grad_norm
                    # We scale perturbation relative to the weight magnitude
                    weight_norm = torch.norm(param.data)
                    perturbation = (
                        self.adv_eps * weight_norm * param.grad / (grad_norm + e)
                    )
                    param.data.add_(perturbation)

    def restore(self):
        """
        Restores the original model weights from backup.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def get_optimizer_grouped_parameters(model):
    """
    Constructs the parameter groups for the optimizer with Layer-Wise Learning Rate Decay (LLRD).

    Strategy:
    1. 'Head' parameters (Classifier, Pooler) get `Config.head_lr`.
    2. Transformer layers get decaying LRs: Top layer gets `Config.lr`,
       lower layers get `Config.lr * (decay ** depth)`.
    3. Embeddings get the lowest LR.
    4. Weight decay is applied to weights but not biases or LayerNorms.

    Args:
        model (nn.Module): The model to optimize.

    Returns:
        list: A list of dictionaries defining parameter groups.
    """
    # Define exclusion list for weight decay
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # Initialize parameter groups
    optimizer_grouped_parameters = []

    # Identify model structure for LLRD
    # DeBERTa V3 Large typically has 24 layers.
    # We access the config to be dynamic, or fallback to standard assumption.
    if hasattr(model, "config"):
        num_layers = model.config.num_hidden_layers
    else:
        # Fallback for DeBERTa-large if config not directly accessible on wrapper
        num_layers = 24

    # Organize parameters
    # We need to map parameter names to their specific LR

    # 1. Collect all named parameters
    named_parameters = list(model.named_parameters())

    # 2. Define Learning Rates per layer
    # Layer ID mapping:
    #   - embeddings: 0
    #   - encoder.layer.0: 1
    #   ...
    #   - encoder.layer.23: 24
    #   - head (pooler/fc): 25 (handled separately)

    # Base LLRD calculation
    # We want the top transformer layer to have Config.lr
    # Lower layers decay by Config.llrd_decay

    for name, param in named_parameters:
        if not param.requires_grad:
            continue

        # Determine Weight Decay
        if any(nd in name for nd in no_decay):
            wd = 0.0
        else:
            wd = Config.weight_decay

        # Determine Learning Rate
        lr = Config.lr

        if "embeddings" in name:
            # Embeddings get the strongest decay (furthest from output)
            lr = Config.lr * (Config.llrd_decay**num_layers)

        elif "encoder.layer" in name:
            # Extract layer index
            # format: model.encoder.layer.15.output...
            try:
                # Split by dot and find the integer after 'layer'
                parts = name.split(".")
                layer_idx = -1
                for i, part in enumerate(parts):
                    if part == "layer":
                        layer_idx = int(parts[i + 1])
                        break

                if layer_idx >= 0:
                    # Depth from top: (num_layers - 1) is top, 0 is bottom
                    # We want top to be decay^0, bottom to be decay^(num_layers-1)
                    distance_from_top = num_layers - 1 - layer_idx
                    lr = Config.lr * (Config.llrd_decay**distance_from_top)
            except (ValueError, IndexError):
                # Fallback if parsing fails
                lr = Config.lr

        elif "pooler" in name or "fc" in name or "classifier" in name:
            # Task-specific head gets higher LR
            lr = Config.head_lr

        # Append to group
        optimizer_grouped_parameters.append(
            {"params": [param], "weight_decay": wd, "lr": lr}
        )

    return optimizer_grouped_parameters
