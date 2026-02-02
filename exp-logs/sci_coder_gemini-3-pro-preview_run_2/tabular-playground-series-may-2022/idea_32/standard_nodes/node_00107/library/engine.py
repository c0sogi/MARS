import math
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def get_optimizer_params(model, weight_decay_encoder, weight_decay_bias):
    """
    Separates model parameters into two groups for decoupled weight decay:
    1. Decay Group: Weights of Linear, Embedding, and Attention layers.
    2. No-Decay Group: Biases, LayerNorm parameters, and Positional Embeddings.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Identify parameters that should not decay
        # - 1D parameters (biases, scale factors)
        # - Explicit bias terms
        # - Normalization layer parameters
        # - Positional embeddings
        if (
            param.ndim <= 1
            or name.endswith(".bias")
            or "norm" in name
            or "pos_embed" in name
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay_encoder},
        {"params": no_decay_params, "weight_decay": weight_decay_bias},
    ]


def initialize_weights(model):
    """
    Applies specific initialization schemes to the model:
    - Transformer Encoder: Xavier (Glorot) Uniform
    - SwiGLU Stems & Backbone: Kaiming (He) Uniform
    - Positional Embeddings: Normal(0, 0.02)
    - Output Head: Xavier Uniform
    """
    # 1. Initialize Transformer Encoder (Xavier)
    if hasattr(model, "transformer_encoder"):
        for p in model.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # 2. Initialize Positional Embeddings (Normal)
    for name, module in model.named_modules():
        # Check for the specific class name from modules.py
        if "LearnablePositionalEncoding" in str(type(module)):
            if hasattr(module, "pos_embed"):
                nn.init.normal_(module.pos_embed, mean=0.0, std=0.02)

    # 3. Initialize Linear Layers (Kaiming for Backbone, Xavier for Head)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Skip if inside transformer (already handled)
            if "transformer_encoder" in name:
                continue

            if "head" in name:
                # Output head uses Xavier
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                # SwiGLU Stems and Backbone use Kaiming Uniform
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    # Uniform initialization for bias based on fan_in
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Executes one training epoch.
    Returns the average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = len(dataloader)

    for batch in dataloader:
        # Move data to device
        continuous = batch["continuous"].to(device)
        sequence = batch["sequence"].to(device)
        targets = batch["target"].to(device).unsqueeze(1)

        optimizer.zero_grad()

        # Forward pass
        logits = model(continuous, sequence)
        loss = criterion(logits, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on a validation set.
    Returns tuple: (average_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            continuous = batch["continuous"].to(device)
            sequence = batch["sequence"].to(device)
            targets = batch["target"].to(device)

            # Forward pass
            logits = model(continuous, sequence)

            # Calculate loss (unsqueeze targets to match logits shape (B, 1))
            loss = criterion(logits, targets.unsqueeze(1))
            running_loss += loss.item()

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(logits).squeeze(1)

            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0

    # Calculate AUC
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle cases where only one class is present in the batch/set
        auc = 0.5

    return avg_loss, auc
