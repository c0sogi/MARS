import torch
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import compute_auc
from library.dataset import BirdDataset


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: torch.nn.Module,
    device: torch.device,
    mixup_alpha: float = Config.MIXUP_ALPHA,
):
    """
    Standard training loop for Anchor models (Phase 1).
    Supports Mixup augmentation.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        if mixup_alpha > 0:
            # Generate Mixup lambda
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            index = torch.randperm(batch_size).to(device)

            # Mix images
            mixed_images = lam * images + (1 - lam) * images[index]

            # Mix targets
            targets_a, targets_b = targets, targets[index]

            # Forward pass
            logits = model(mixed_images)

            # Compute mixed loss
            loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(
                logits, targets_b
            )
        else:
            logits = model(images)
            loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def distill_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: torch.nn.Module,
    device: torch.device,
    mixup_alpha: float = Config.MIXUP_ALPHA,
):
    """
    Distillation training loop for Born-Again Ensemble (Phase 3).
    Uses DistillationLoss to learn from both hard targets and TTA-enhanced soft targets.
    Supports Mixup by mixing both sets of targets.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        images = batch["image"].to(device)
        hard_targets = batch["target"].to(device)
        soft_targets = batch["soft_target"].to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        if mixup_alpha > 0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            index = torch.randperm(batch_size).to(device)

            mixed_images = lam * images + (1 - lam) * images[index]

            # Mix both hard and soft targets
            hard_a, hard_b = hard_targets, hard_targets[index]
            soft_a, soft_b = soft_targets, soft_targets[index]

            logits = model(mixed_images)

            # Compute loss against both mixed pairs
            # criterion is expected to be DistillationLoss(logits, hard, soft)
            loss_a = criterion(logits, hard_a, soft_a)
            loss_b = criterion(logits, hard_b, soft_b)
            loss = lam * loss_a + (1 - lam) * loss_b
        else:
            logits = model(images)
            loss = criterion(logits, hard_targets, soft_targets)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def valid_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
):
    """
    Validation loop. Computes Loss and Macro-AUC.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            batch_size = images.size(0)

            logits = model(images)
            loss = criterion(logits, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Store probabilities and targets for AUC
            probs = torch.sigmoid(logits)
            preds_list.append(probs.cpu())
            targets_list.append(targets.cpu())

    epoch_loss = running_loss / dataset_size

    all_preds = torch.cat(preds_list)
    all_targets = torch.cat(targets_list)

    epoch_auc = compute_auc(all_targets, all_preds)

    return epoch_loss, epoch_auc


def tta_inference(
    model: torch.nn.Module,
    images: np.ndarray,
    rec_ids: np.ndarray,
    device: torch.device,
    tta_steps: int = Config.TTA_STEPS,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
) -> np.ndarray:
    """
    Performs Cyclic Test-Time Augmentation (TTA).
    Generates predictions for multiple time-shifts (0%, 25%, 50%, 75%) and averages them.

    Args:
        model: Trained model.
        images: Array of images (N, H, W, 3).
        rec_ids: Array of recording IDs.
        device: Torch device.
        tta_steps: Number of cyclic shifts (default 4).

    Returns:
        np.ndarray: Averaged probability predictions (N, Num_Classes).
    """
    model.eval()
    all_tta_preds = []

    # Iterate through defined TTA steps (0 to tta_steps-1)
    # BirdDataset handles the actual shifting logic via tta_shift parameter
    for step in range(tta_steps):
        dataset = BirdDataset(
            images=images, rec_ids=rec_ids, mode="test", tta_shift=step
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        step_preds = []
        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"].to(device)
                logits = model(imgs)
                probs = torch.sigmoid(logits)
                step_preds.append(probs.cpu().numpy())

        # Concatenate predictions for this TTA step
        all_tta_preds.append(np.concatenate(step_preds, axis=0))

    # Average predictions across all TTA steps
    avg_preds = np.mean(all_tta_preds, axis=0)

    return avg_preds
