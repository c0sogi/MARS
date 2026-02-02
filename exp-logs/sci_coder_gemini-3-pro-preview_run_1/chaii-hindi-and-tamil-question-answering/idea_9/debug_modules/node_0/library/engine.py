import torch
import torch.nn as nn
import numpy as np
from library.model import FGM


def train_fn(data_loader, model, optimizer, device, config, scheduler=None):
    """
    Executes one training epoch with Adversarial Training (FGM) and Mixed Precision.

    Args:
        data_loader: PyTorch DataLoader for training data.
        model: The model to train.
        optimizer: The optimizer.
        device: The device to run on (cuda/cpu).
        config: Configuration object containing hyperparameters.
        scheduler: Learning rate scheduler (optional).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()

    # Initialize loss functions
    # CrossEntropyLoss for span indices (start/end)
    loss_fct_span = nn.CrossEntropyLoss()
    # BCEWithLogitsLoss for binary relevance classification
    loss_fct_relevance = nn.BCEWithLogitsLoss()

    # Initialize scaler for Automatic Mixed Precision (AMP)
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

    # Initialize FGM for adversarial training if enabled
    if config.use_fgm:
        fgm = FGM(model, epsilon=config.fgm_epsilon, param_name=config.fgm_name)

    losses = []

    for batch_idx, batch in enumerate(data_loader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)
        relevance_labels = batch["relevance_labels"].to(device)

        # --- Standard Forward Pass ---
        with torch.cuda.amp.autocast(enabled=config.use_amp):
            start_logits, end_logits, relevance_logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            )

            # Calculate Losses
            start_loss = loss_fct_span(start_logits, start_positions)
            end_loss = loss_fct_span(end_logits, end_positions)
            relevance_loss = loss_fct_relevance(relevance_logits, relevance_labels)

            # Weighted sum of losses
            loss = (start_loss + end_loss) / 2 + config.aux_weight * relevance_loss

        # --- Standard Backward Pass ---
        scaler.scale(loss).backward()

        # --- Adversarial Training (FGM) ---
        if config.use_fgm:
            # Perturb embeddings
            fgm.attack()

            with torch.cuda.amp.autocast(enabled=config.use_amp):
                # Adversarial Forward Pass
                start_logits_adv, end_logits_adv, relevance_logits_adv = model(
                    input_ids=input_ids, attention_mask=attention_mask
                )

                # Calculate Adversarial Losses
                start_loss_adv = loss_fct_span(start_logits_adv, start_positions)
                end_loss_adv = loss_fct_span(end_logits_adv, end_positions)
                relevance_loss_adv = loss_fct_relevance(
                    relevance_logits_adv, relevance_labels
                )

                loss_adv = (
                    start_loss_adv + end_loss_adv
                ) / 2 + config.aux_weight * relevance_loss_adv

            # Adversarial Backward Pass
            scaler.scale(loss_adv).backward()

            # Restore original embeddings
            fgm.restore()

        # --- Optimization Step ---
        # Unscale gradients before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())

    return np.mean(losses)


def predict_fn(data_loader, model, device):
    """
    Generates predictions for the test set.

    Args:
        data_loader: PyTorch DataLoader for test data.
        model: The trained model.
        device: The device to run on.

    Returns:
        tuple: (start_logits, end_logits, relevance_logits) as numpy arrays.
    """
    model.eval()

    start_logits_list = []
    end_logits_list = []
    relevance_logits_list = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            start_logits, end_logits, relevance_logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            )

            # Move to CPU and convert to numpy
            start_logits_list.append(start_logits.cpu().numpy())
            end_logits_list.append(end_logits.cpu().numpy())
            relevance_logits_list.append(relevance_logits.cpu().numpy())

    # Concatenate all batches
    start_preds = np.concatenate(start_logits_list, axis=0)
    end_preds = np.concatenate(end_logits_list, axis=0)
    relevance_preds = np.concatenate(relevance_logits_list, axis=0)

    return start_preds, end_preds, relevance_preds
