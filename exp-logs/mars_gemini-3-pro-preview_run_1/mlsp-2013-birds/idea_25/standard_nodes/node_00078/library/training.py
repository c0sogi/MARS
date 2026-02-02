import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from library.utils import save_checkpoint, moving_average, update_bn
from library.data import mixup_data


def train_one_epoch(model, loader, criterion, optimizer, device, mixup_alpha=0.2):
    """
    Trains the model for one epoch using Input-Level Mixup.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function (e.g., BCEWithLogitsLoss).
        optimizer (Optimizer): Optimizer.
        device (str): Device to run training on.
        mixup_alpha (float): Alpha parameter for Beta distribution in Mixup.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, targets, _ in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_size = inputs.size(0)

        # Apply Mixup
        inputs, targets_a, targets_b, lam = mixup_data(
            inputs, targets, alpha=mixup_alpha, device=device
        )

        # Forward pass
        outputs = model(inputs)

        # Compute Loss
        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
            outputs, targets_b
        )

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (str): Device to run evaluation on.

    Returns:
        dict: Dictionary containing 'loss' and 'auc'.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = inputs.size(0)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for ROC AUC calculation
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    avg_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)

    # Compute ROC AUC
    # Handle potential edge cases where a class might not be present in the validation batch
    # Cite debug_lesson_5: Safeguard Global Metrics Against Degenerate Data Subsets
    aucs = []
    for i in range(all_targets.shape[1]):
        try:
            if len(np.unique(all_targets[:, i])) == 2:
                aucs.append(roc_auc_score(all_targets[:, i], all_preds[:, i]))
        except ValueError:
            pass

    if len(aucs) > 0:
        auc_score = np.mean(aucs)
    else:
        auc_score = 0.5

    return {"loss": avg_loss, "auc": auc_score}


def run_training_schedule(
    model,
    train_loader,
    val_loader,
    epochs=50,
    swa_start_epoch_pct=0.75,
    lr=1e-3,
    device="cuda",
    checkpoint_dir="./working/idea_25/checkpoints",
):
    """
    Orchestrates the training schedule, including Optimizer, Scheduler, and SWA.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training loader.
        val_loader (DataLoader): Validation loader.
        epochs (int): Total number of epochs.
        swa_start_epoch_pct (float): Fraction of epochs before starting SWA (e.g., 0.75).
        lr (float): Initial learning rate.
        device (str): Device.
        checkpoint_dir (str): Directory to save checkpoints.

    Returns:
        tuple: (best_model_state, swa_model_state)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc = 0.0
    best_epoch = 0

    # SWA Setup
    swa_model = None
    swa_start_epoch = int(epochs * swa_start_epoch_pct)
    swa_n = 0

    print(
        f"Starting training for {epochs} epochs. SWA starts at epoch {swa_start_epoch}."
    )

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate (Base Model)
        val_metrics = validate(model, val_loader, criterion, device)
        val_loss = val_metrics["loss"]
        val_auc = val_metrics["auc"]

        # Update Scheduler
        scheduler.step()

        # SWA Logic
        if epoch >= swa_start_epoch:
            if swa_model is None:
                swa_model = torch.optim.swa_utils.AveragedModel(model)
            else:
                swa_model.update_parameters(model)
            swa_n += 1

        # Checkpointing (Base Model)
        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc
            best_epoch = epoch

        # Save current state
        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_metric": best_auc,
            },
            is_best=is_best,
            checkpoint_dir=checkpoint_dir,
            filename="model_last.pth",
        )

        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.10f} | Best AUC: {best_auc:.10f} @ Epoch {best_epoch}"
        )

    # Finalize SWA
    if swa_model is not None:
        print("Updating SWA BatchNorm statistics...")
        # update_bn expects a standard model, swa_model is AveragedModel wrapper
        # We need to pass the underlying module or use torch.optim.swa_utils.update_bn
        # The library.utils.update_bn is a custom implementation.
        # Let's use the custom one on the module.
        update_bn(train_loader, swa_model.module, device=device)

        # Validate SWA
        swa_metrics = validate(swa_model.module, val_loader, criterion, device)
        print(
            f"SWA Final Results | Val Loss: {swa_metrics['loss']:.6f} | Val AUC: {swa_metrics['auc']:.10f}"
        )

        save_checkpoint(
            {
                "epoch": epochs,
                "state_dict": swa_model.module.state_dict(),
                "best_metric": swa_metrics["auc"],
            },
            is_best=False,
            checkpoint_dir=checkpoint_dir,
            filename="model_swa.pth",
        )
        return model, swa_model.module

    return model, None
