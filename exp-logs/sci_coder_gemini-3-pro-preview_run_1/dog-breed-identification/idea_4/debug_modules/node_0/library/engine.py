import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import log_loss
from typing import Tuple, List, Dict

from library.config import Config
from library.utils import log_message, print_metric


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: Training data loader.
        optimizer: The optimizer.
        device: The device to run on.
        epoch: Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.CrossEntropyLoss()

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: Validation data loader.
        device: The device to run on.

    Returns:
        Tuple[float, np.ndarray, np.ndarray]: (Log Loss, Predictions (Probs), True Labels)
    """
    model.eval()
    all_preds = []
    all_labels = []

    # Softmax for probability conversion
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = softmax(outputs)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    predictions = np.concatenate(all_preds)
    true_labels = np.concatenate(all_labels)

    # Calculate Log Loss
    # labels are indices, log_loss expects them or one-hot.
    # Providing labels as list of indices works if labels covers all classes or we provide 'labels' arg.
    # To be safe and explicit with 120 classes:
    metric = log_loss(true_labels, predictions, labels=list(range(Config.NUM_CLASSES)))

    return metric, predictions, true_labels


def predict_tta(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[List[str], np.ndarray]:
    """
    Generates predictions for the test set using Test Time Augmentation (TTA).
    Averages predictions from original image and horizontally flipped image.

    Args:
        model: The PyTorch model.
        dataloader: Test data loader (returns image, id).
        device: The device to run on.

    Returns:
        Tuple[List[str], np.ndarray]: (List of IDs, Array of Probabilities)
    """
    model.eval()
    all_ids = []
    all_probs = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for images, ids in dataloader:
            images = images.to(device)

            # 1. Forward pass original
            output_orig = model(images)
            probs_orig = softmax(output_orig)

            # 2. Forward pass flipped (TTA)
            if Config.USE_TTA:
                # Flip along width (dimension 3: N, C, H, W)
                images_flipped = torch.flip(images, dims=[3])
                output_flipped = model(images_flipped)
                probs_flipped = softmax(output_flipped)

                # Average probabilities
                probs_avg = (probs_orig + probs_flipped) / 2.0
            else:
                probs_avg = probs_orig

            all_probs.append(probs_avg.cpu().numpy())
            all_ids.extend(ids)

    final_probs = np.concatenate(all_probs)
    return all_ids, final_probs


def train_loop(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    fold_idx: int,
    device: torch.device,
) -> None:
    """
    Orchestrates the two-phase training process for a single fold.
    Phase 1: Frozen Backbone, Train Head.
    Phase 2: Unfrozen Backbone, Discriminative Learning Rates.
    Handles Early Stopping and Model Checkpointing.

    Args:
        model: The model instance.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        fold_idx: Index of the current fold.
        device: Computation device.
    """
    best_loss = float("inf")
    patience_counter = 0
    save_path = Config.get_model_path(fold_idx)

    # ==========================================
    # Phase 1: Head Adaptation
    # ==========================================
    log_message(
        f"\n[Fold {fold_idx}] Starting Phase 1: Head Adaptation (Frozen Backbone)"
    )

    # Freeze backbone
    model.freeze_backbone(freeze=True)

    # Optimizer for Head only
    # We can just pass all parameters; those with requires_grad=False will be ignored by optimizer usually,
    # but explicitly filtering is cleaner or using the helper method with 0 LR for backbone (though freeze handles grad).
    # Ideally, we just optimize head params.
    head_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer_p1 = torch.optim.AdamW(
        head_params, lr=Config.LR_HEAD_PHASE_1, weight_decay=Config.WEIGHT_DECAY
    )

    for epoch in range(1, Config.EPOCHS_PHASE_1 + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer_p1, device, epoch)
        val_loss, _, _ = evaluate(model, val_loader, device)

        log_message(
            f"Phase 1 - Epoch {epoch}/{Config.EPOCHS_PHASE_1} - Train Loss: {train_loss:.4f}"
        )
        print_metric("Phase 1 - Val Log Loss", val_loss)

        # Checkpoint if best
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            # log_message(f"New best model saved at {save_path}")
            patience_counter = 0
        else:
            patience_counter += 1

    # ==========================================
    # Phase 2: Fine-Tuning
    # ==========================================
    log_message(
        f"\n[Fold {fold_idx}] Starting Phase 2: Fine-Tuning (Unfrozen Backbone)"
    )

    # Unfreeze backbone
    model.freeze_backbone(freeze=False)

    # Optimizer with Discriminative Learning Rates
    param_groups = model.get_optimizer_params(
        lr_backbone=Config.LR_BACKBONE, lr_head=Config.LR_HEAD_PHASE_2
    )
    optimizer_p2 = torch.optim.AdamW(param_groups, weight_decay=Config.WEIGHT_DECAY)

    # Reset patience for Phase 2? Usually yes, or continue.
    # Given the shift in dynamics, giving it a fresh patience count is safer.
    patience_counter = 0

    # Load best weights from Phase 1 to ensure we start Phase 2 from the best point so far?
    # Or continue from last state?
    # Standard practice: Continue from last state to avoid losing momentum,
    # but ensure 'best_loss' is carried over.
    # However, if Phase 1 overfitted at the end, reloading best might be better.
    # Let's reload best to be safe.
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        # log_message("Loaded best model from Phase 1 for Phase 2 initialization.")

    for epoch in range(1, Config.EPOCHS_PHASE_2 + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer_p2, device, epoch)
        val_loss, _, _ = evaluate(model, val_loader, device)

        log_message(
            f"Phase 2 - Epoch {epoch}/{Config.EPOCHS_PHASE_2} - Train Loss: {train_loss:.4f}"
        )
        print_metric("Phase 2 - Val Log Loss", val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            # log_message(f"New best model saved at {save_path}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            log_message(f"Early stopping triggered at epoch {epoch} of Phase 2.")
            break

    log_message(f"[Fold {fold_idx}] Training completed. Best Val Log Loss: {best_loss}")
