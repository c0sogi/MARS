import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for all relevant libraries to ensure reproducibility.

    Args:
        seed (int): The random seed to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, path):
    """
    Saves the model's state dictionary to the specified path.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): The file path where the checkpoint will be saved.
    """
    # Ensure the directory exists
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    torch.save(model.state_dict(), path)


def get_optimizer_params(model, learning_rate, weight_decay, llrd_decay):
    """
    Constructs the parameter groups for the optimizer with Layer-Wise Learning Rate Decay (LLRD).

    This function groups parameters based on their location in the RoBERTa-Large model:
    - Head parameters: Base learning_rate
    - Encoder layers: learning_rate * (llrd_decay ** distance_from_head)
    - Embeddings: Lowest learning_rate

    It also applies weight decay to weights but excludes biases and LayerNorm parameters.

    Args:
        model (torch.nn.Module): The model to optimize.
        learning_rate (float): The base learning rate (applied to the head).
        weight_decay (float): The weight decay coefficient.
        llrd_decay (float): The decay factor for lower layers (e.g., 0.95).

    Returns:
        list: A list of dictionaries containing parameter groups for the optimizer.
    """
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

    # RoBERTa-Large has 24 layers.
    # We assume standard naming conventions: 'roberta.embeddings', 'roberta.encoder.layer.X'
    num_layers = 24

    # Dictionary to group parameters by (lr, weight_decay) key
    # Key: (learning_rate, weight_decay) -> Value: list of parameters
    param_groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 1. Determine Learning Rate based on depth
        if "embeddings" in name:
            # Embeddings are at the bottom: decay^(num_layers + 1)
            lr = learning_rate * (llrd_decay ** (num_layers + 1))

        elif "encoder.layer" in name:
            # Extract layer index to calculate depth
            # name format example: "roberta.encoder.layer.11.output.dense.weight"
            parts = name.split(".")
            layer_idx = -1

            # Find the index immediately following "layer"
            for i, part in enumerate(parts):
                if part == "layer" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_idx = int(parts[i + 1])
                    break

            if layer_idx != -1:
                # Layer 23 (top) is distance 1 from head -> decay^1
                # Layer 0 (bottom) is distance 24 from head -> decay^24
                distance_from_head = num_layers - layer_idx
                lr = learning_rate * (llrd_decay**distance_from_head)
            else:
                # Fallback if parsing fails, treat as generic backbone
                lr = learning_rate * llrd_decay

        else:
            # Head parameters or other top-level components
            lr = learning_rate

        # 2. Determine Weight Decay
        if any(nd in name for nd in no_decay):
            wd = 0.0
        else:
            wd = weight_decay

        # 3. Add to groups
        group_key = (lr, wd)
        if group_key not in param_groups:
            param_groups[group_key] = []
        param_groups[group_key].append(param)

    # Format the groups for the optimizer
    optimizer_params = []
    for (lr, wd), params in param_groups.items():
        optimizer_params.append({"params": params, "lr": lr, "weight_decay": wd})

    return optimizer_params
