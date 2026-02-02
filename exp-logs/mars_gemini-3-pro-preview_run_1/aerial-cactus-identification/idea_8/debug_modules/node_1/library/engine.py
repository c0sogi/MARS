import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from library.dataset import mixup_data, mixup_criterion


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch,
    mixup_alpha=0.2,
    aux_weight=0.4,
):
    """
    Trains the model for one epoch using SAM (Sharpness-Aware Minimization) and Mixup.

    Args:
        model: The PyTorch model (CactusRepVGG).
        dataloader: Training dataloader.
        criterion: Loss function (e.g., BCEWithLogitsLoss).
        optimizer: SAM optimizer instance.
        device: 'cuda' or 'cpu'.
        epoch: Current epoch number (for logging).
        mixup_alpha: Alpha parameter for Mixup (Beta distribution).
        aux_weight: Weight for the auxiliary head loss.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device).view(-1, 1)

        batch_size = inputs.size(0)

        # --- 1. Mixup Preparation ---
        inputs_mixed, targets_a, targets_b, lam = mixup_data(
            inputs, targets, alpha=mixup_alpha, use_cuda=(device.type == "cuda")
        )

        # --- 2. First Forward-Backward Pass (Clean weights) ---
        # Model returns (main_output, aux_output) in training mode
        main_out, aux_out = model(inputs_mixed)

        loss_main = mixup_criterion(criterion, main_out, targets_a, targets_b, lam)
        loss_aux = mixup_criterion(criterion, aux_out, targets_a, targets_b, lam)
        loss = loss_main + (aux_weight * loss_aux)

        loss.backward()

        # SAM: Save weights, apply perturbation
        optimizer.first_step(zero_grad=True)

        # --- 3. Second Forward-Backward Pass (Perturbed weights) ---
        # Re-compute outputs with perturbed weights
        main_out_2, aux_out_2 = model(inputs_mixed)

        loss_main_2 = mixup_criterion(criterion, main_out_2, targets_a, targets_b, lam)
        loss_aux_2 = mixup_criterion(criterion, aux_out_2, targets_a, targets_b, lam)
        loss_2 = loss_main_2 + (aux_weight * loss_aux_2)

        loss_2.backward()

        # SAM: Restore weights, apply base optimizer step
        optimizer.second_step(zero_grad=True)

        # Update stats (using the first loss as the reported metric)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    print(f"Epoch {epoch} Training Loss: {epoch_loss:.6f}")

    return epoch_loss


def validate_one_epoch(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: Validation dataloader.
        criterion: Loss function.
        device: 'cuda' or 'cpu'.

    Returns:
        dict: Dictionary containing 'val_loss' and 'val_auc'.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            batch_size = inputs.size(0)

            # Model returns only main_output in eval mode
            outputs = model(inputs)

            # Since targets might be (N,) and outputs (N, 1), ensure shapes match for BCE
            loss = criterion(outputs, targets.view_as(outputs))

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Collect predictions for AUC
            # Apply sigmoid to logits to get probabilities
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate all batches
    all_preds = np.concatenate(all_preds).ravel()
    all_targets = np.concatenate(all_targets).ravel()

    # Calculate ROC AUC
    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle edge case if only one class is present in a small validation batch
        auc_score = 0.5

    print(f"Validation Loss: {epoch_loss:.10f} | AUC: {auc_score:.10f}")

    return {"val_loss": epoch_loss, "val_auc": auc_score}


def save_checkpoint(model, output_dir, filename):
    """
    Saves the model state dict.

    Args:
        model: PyTorch model.
        output_dir: Directory to save the file.
        filename: Name of the file.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    torch.save(model.state_dict(), save_path)
    # print(f"Model saved to {save_path}")
