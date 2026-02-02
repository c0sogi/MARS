import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    Ensures deterministic behavior for CuDNN backends.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FGM:
    """
    Fast Gradient Method (FGM) for Adversarial Training.
    Perturbs embeddings based on gradients to improve model robustness.
    """

    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name="word_embeddings"):
        """
        Applies perturbation to the embeddings.

        Args:
            epsilon (float): Magnitude of the perturbation.
            emb_name (str): Substring to identify embedding parameters.
        """
        for name, param in self.model.named_parameters():
            # Apply only to parameters that require gradients and match the embedding name
            if param.requires_grad and emb_name in name and param.grad is not None:
                # Save original data
                self.backup[name] = param.data.clone()

                # Calculate norm of the gradient
                norm = torch.norm(param.grad)

                # Apply perturbation if norm is valid (avoid division by zero)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name="word_embeddings"):
        """
        Restores the original embeddings from backup.

        Args:
            emb_name (str): Substring to identify embedding parameters.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}


def get_optimizer_grouped_parameters(model, config):
    """
    Configures layer-wise learning rate decay (LLRD) and weight decay.

    Strategy:
    1. Identify the depth of the encoder (e.g., 24 layers for XLM-R Large).
    2. Assign Layer Indices: Embeddings=0, Layer 0=1, ..., Head=Max+2.
    3. Apply exponential decay: LR_layer = Base_LR * (Decay ^ (Head_Idx - Layer_Idx)).
    4. Apply weight decay to ALL parameters (including bias/LayerNorm) to prevent overfitting.
    """

    # 1. Identify Model Depth dynamically
    max_encoder_layer = 0
    for name, _ in model.named_parameters():
        if "encoder.layer" in name:
            try:
                # Expected format: ...encoder.layer.N...
                parts = name.split(".")
                # Find the part after 'layer'
                if "layer" in parts:
                    idx = int(parts[parts.index("layer") + 1])
                    if idx > max_encoder_layer:
                        max_encoder_layer = idx
            except (ValueError, IndexError):
                continue

    # Define Layer Indices
    # Embeddings: 0
    # Encoder Layer i: i + 1
    # Head/Top: max_encoder_layer + 2 (conceptually above the last encoder layer)
    head_layer_idx = max_encoder_layer + 2

    # 2. Group Parameters by Layer
    param_groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine layer index for this parameter
        if "embeddings" in name and "encoder" not in name:
            layer_idx = 0
        elif "encoder.layer" in name:
            try:
                parts = name.split(".")
                if "layer" in parts:
                    idx = int(parts[parts.index("layer") + 1])
                    layer_idx = idx + 1
                else:
                    layer_idx = 0  # Fallback
            except:
                layer_idx = 0  # Fallback
        else:
            # Parameters not in embeddings or encoder layers are considered Head/Top
            # This includes the task-specific heads (span, relevance) and poolers
            layer_idx = head_layer_idx

        if layer_idx not in param_groups:
            param_groups[layer_idx] = []
        param_groups[layer_idx].append(param)

    # 3. Create Optimizer List with LLRD
    optimizer_grouped_parameters = []

    base_lr = config.LEARNING_RATE
    decay_factor = config.LLRD_DECAY
    weight_decay = config.WEIGHT_DECAY

    # Iterate through all identified layers to assign LRs
    for layer_idx in range(head_layer_idx + 1):
        if layer_idx in param_groups:
            # Calculate Learning Rate
            # Formula: LR = Base_LR * (Decay ^ (Head_Idx - Current_Idx))
            # Head (idx=Head_Idx) -> Power 0 -> Base_LR
            # Embeddings (idx=0) -> Power Head_Idx -> Base_LR * Decay^Head_Idx

            exponent = head_layer_idx - layer_idx
            layer_lr = base_lr * (decay_factor**exponent)

            optimizer_grouped_parameters.append(
                {
                    "params": param_groups[layer_idx],
                    "lr": layer_lr,
                    "weight_decay": weight_decay,  # Applied to all parameters as per instructions
                }
            )

    return optimizer_grouped_parameters
