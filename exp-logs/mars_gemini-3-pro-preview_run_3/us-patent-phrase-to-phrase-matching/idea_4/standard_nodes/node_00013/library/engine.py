import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import compute_metrics, get_logger


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    """
    Constructs the parameter groups for the optimizer with Layer-wise Learning Rate Decay (LLRD).

    Args:
        model (nn.Module): The model to optimize.
        encoder_lr (float): Base learning rate for the top transformer layer.
        decoder_lr (float): Learning rate for the custom heads (decoder).
        weight_decay (float): Weight decay coefficient.

    Returns:
        list: A list of dictionaries defining parameter groups.
    """
    # Define parameter groups
    # 1. Embeddings
    # 2. Encoder Layers (0 to N-1)
    # 3. Decoder/Heads (Pooler, FC layers)

    # Access the backbone config to get number of layers
    # CustomModel -> model (AutoModel) -> config
    num_hidden_layers = model.config.num_hidden_layers

    # Initialize groups
    optimizer_parameters = []

    # We will iterate through named parameters and assign them to specific groups
    # Map parameter names to specific LRs

    # Calculate LR for each layer: lr = encoder_lr * (decay ** depth)
    # Depth 0 is the last layer (closest to output), Depth N is embeddings

    # Helper to determine if parameter should have weight decay
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # 1. Capture Decoder/Head parameters (High LR)
    head_params = (
        list(model.pooler.named_parameters())
        + list(model.fc_regressor.named_parameters())
        + list(model.fc_classifier.named_parameters())
    )

    head_group_decay = []
    head_group_no_decay = []

    for name, param in head_params:
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            head_group_no_decay.append(param)
        else:
            head_group_decay.append(param)

    optimizer_parameters.append(
        {"params": head_group_decay, "lr": decoder_lr, "weight_decay": weight_decay}
    )
    optimizer_parameters.append(
        {"params": head_group_no_decay, "lr": decoder_lr, "weight_decay": 0.0}
    )

    # 2. Capture Backbone Layers (Decaying LR)
    # Deberta V3 structure: model.model.encoder.layer.{i}
    # We iterate backwards from last layer to 0

    for layer_i in range(num_hidden_layers - 1, -1, -1):
        layer_lr = encoder_lr * (Config.llrd_decay ** (num_hidden_layers - 1 - layer_i))

        # Get parameters for this specific layer
        # Note: model.model is the AutoModel inside CustomModel
        layer_params = list(model.model.encoder.layer[layer_i].named_parameters())

        layer_group_decay = []
        layer_group_no_decay = []

        for name, param in layer_params:
            if not param.requires_grad:
                continue
            if any(nd in name for nd in no_decay):
                layer_group_no_decay.append(param)
            else:
                layer_group_decay.append(param)

        optimizer_parameters.append(
            {"params": layer_group_decay, "lr": layer_lr, "weight_decay": weight_decay}
        )
        optimizer_parameters.append(
            {"params": layer_group_no_decay, "lr": layer_lr, "weight_decay": 0.0}
        )

    # 3. Capture Embeddings and Relative Position Bias (Lowest LR)
    # Embeddings LR = encoder_lr * (decay ** num_layers)
    embedding_lr = encoder_lr * (Config.llrd_decay**num_hidden_layers)

    # Collect all remaining parameters (embeddings, final layer norm of encoder if any, etc.)
    # We've already collected heads and encoder layers.
    # The safest way is to check what's left or target embeddings specifically.
    # Deberta V3: model.embeddings

    embedding_params = list(model.model.embeddings.named_parameters())
    # Also include relative_attention bias if present in encoder but not in layers
    # For Deberta, relative embeddings might be separate.

    # To be safe, we can iterate all params and check if they were already added.
    # However, explicitly grabbing embeddings is standard.

    emb_group_decay = []
    emb_group_no_decay = []

    for name, param in embedding_params:
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            emb_group_no_decay.append(param)
        else:
            emb_group_decay.append(param)

    optimizer_parameters.append(
        {"params": emb_group_decay, "lr": embedding_lr, "weight_decay": weight_decay}
    )
    optimizer_parameters.append(
        {"params": emb_group_no_decay, "lr": embedding_lr, "weight_decay": 0.0}
    )

    return optimizer_parameters


def train_fn(train_loader, model, optimizer, epoch, scheduler, device, awp=None):
    """
    Executes one training epoch.

    Args:
        train_loader: DataLoader for training data.
        model: The neural network model.
        optimizer: The optimizer.
        epoch (int): Current epoch index (0-based).
        scheduler: Learning rate scheduler.
        device: Torch device.
        awp (AWP, optional): Adversarial Weight Perturbation object.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()

    # Loss functions
    mse_criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    count = 0

    # Iterate over batches
    for step, batch in enumerate(train_loader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets_score = batch["score"].to(device)
        targets_label = batch["label_idx"].to(device)

        batch_size = input_ids.size(0)

        # 1. Forward Pass
        outputs = model(input_ids, attention_mask)

        # Outputs contain "score" (B, 1) and "logits" (B, num_classes)
        pred_score = outputs["score"].view(-1)
        pred_logits = outputs["logits"]

        # Calculate Combined Loss
        loss_mse = mse_criterion(pred_score, targets_score)
        loss_ce = ce_criterion(pred_logits, targets_label)

        loss = (Config.mse_weight * loss_mse) + (Config.ce_weight * loss_ce)

        # 2. Backward Pass
        loss.backward()

        # 3. Adversarial Weight Perturbation (AWP)
        # Only apply after start epoch
        if awp is not None and epoch >= Config.awp_start_epoch:
            awp.attack()

            # Forward pass with perturbed weights
            adv_outputs = model(input_ids, attention_mask)
            adv_pred_score = adv_outputs["score"].view(-1)
            adv_pred_logits = adv_outputs["logits"]

            adv_loss_mse = mse_criterion(adv_pred_score, targets_score)
            adv_loss_ce = ce_criterion(adv_pred_logits, targets_label)

            adv_loss = (Config.mse_weight * adv_loss_mse) + (
                Config.ce_weight * adv_loss_ce
            )

            # Backward pass for adversarial loss
            adv_loss.backward()

            # Restore original weights
            awp._restore()

        # 4. Optimization Step
        nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item() * batch_size
        count += batch_size

    return total_loss / count


def valid_fn(valid_loader, model, device):
    """
    Executes validation loop.

    Args:
        valid_loader: DataLoader for validation data.
        model: The neural network model.
        device: Torch device.

    Returns:
        tuple: (average_loss, pearson_score)
    """
    model.eval()

    mse_criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    count = 0

    preds_list = []
    labels_list = []

    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets_score = batch["score"].to(device)
            targets_label = batch["label_idx"].to(device)

            batch_size = input_ids.size(0)

            outputs = model(input_ids, attention_mask)

            pred_score = outputs["score"].view(-1)
            pred_logits = outputs["logits"]

            loss_mse = mse_criterion(pred_score, targets_score)
            loss_ce = ce_criterion(pred_logits, targets_label)

            loss = (Config.mse_weight * loss_mse) + (Config.ce_weight * loss_ce)

            total_loss += loss.item() * batch_size
            count += batch_size

            # Store predictions for metric computation
            # We clip predictions to [0, 1] for the metric calculation logic if needed,
            # but usually raw scores are fine for Pearson.
            # However, for final submission we clip. Let's keep raw here for correlation.
            preds_list.append(pred_score.cpu().numpy())
            labels_list.append(targets_score.cpu().numpy())

    avg_loss = total_loss / count

    predictions = np.concatenate(preds_list)
    labels = np.concatenate(labels_list)

    # Compute Pearson Correlation
    metrics = compute_metrics(predictions, labels)

    return avg_loss, metrics["pearson"]
