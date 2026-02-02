import torch
import numpy as np
import pandas as pd
import sys
from library.config import Config
from library.utils import get_score
from library.sam import SAM
from library.loss import AnchorDistillationLoss


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device, epoch):
    """
    Performs one epoch of training.
    Handles Mixup, SAM optimization, and Distillation.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    is_sam = isinstance(optimizer, SAM)

    for batch_idx, (images, hard_targets, soft_targets, _) in enumerate(loader):
        images = images.to(device)
        hard_targets = hard_targets.to(device)
        soft_targets = soft_targets.to(device)

        batch_size = images.size(0)

        # --- Mixup Augmentation ---
        use_mixup = Config.MIXUP_ALPHA > 0
        if use_mixup:
            lam = np.random.beta(Config.MIXUP_ALPHA, Config.MIXUP_ALPHA)
            index = torch.randperm(batch_size).to(device)

            mixed_images = lam * images + (1 - lam) * images[index]
            mixed_hard = lam * hard_targets + (1 - lam) * hard_targets[index]
            mixed_soft = lam * soft_targets + (1 - lam) * soft_targets[index]
        else:
            mixed_images = images
            mixed_hard = hard_targets
            mixed_soft = soft_targets

        # --- Forward & Loss Calculation Helper ---
        def compute_loss(input_imgs):
            logits = model(input_imgs)
            if isinstance(loss_fn, AnchorDistillationLoss):
                return loss_fn(logits, mixed_hard, mixed_soft)
            else:
                return loss_fn(logits, mixed_hard)

        # --- Optimization Step ---
        if is_sam:
            # 1. Initial Forward/Backward to compute gradients
            loss = compute_loss(mixed_images)
            loss.backward()

            # 2. SAM Step (Perturb -> Closure -> Update)
            def closure():
                # SAM's first_step zeros grads, so we just need to forward/backward
                l = compute_loss(mixed_images)
                l.backward()
                return l

            optimizer.step(closure)

            # Zero grads is handled inside SAM step or we do it here for safety next iter
            optimizer.zero_grad()

        else:
            # Standard Optimization
            optimizer.zero_grad()
            loss = compute_loss(mixed_images)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        # Step scheduler if it's batch-based (optional check, assuming caller handles epoch-based)
        if scheduler is not None:
            # Simple heuristic: if scheduler has 'step_batch', use it, else assume epoch
            # For this implementation, we assume scheduler is stepped per batch if passed here
            # (common for OneCycleLR). If using ReduceLROnPlateau, pass None here.
            if isinstance(
                scheduler,
                (
                    torch.optim.lr_scheduler.OneCycleLR,
                    torch.optim.lr_scheduler.CyclicLR,
                ),
            ):
                scheduler.step()

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def valid_one_epoch(model, loader, loss_fn, device):
    """
    Performs validation.
    Computes Loss and ROC AUC.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    preds = []
    targets = []

    with torch.no_grad():
        for images, hard_targets, soft_targets, _ in loader:
            images = images.to(device)
            hard_targets = hard_targets.to(device)
            soft_targets = soft_targets.to(device)
            batch_size = images.size(0)

            logits = model(images)

            # Compute loss
            if isinstance(loss_fn, AnchorDistillationLoss):
                loss = loss_fn(logits, hard_targets, soft_targets)
            else:
                loss = loss_fn(logits, hard_targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Store predictions and targets for AUC
            preds.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(hard_targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    auc = get_score(targets, preds)

    return epoch_loss, auc


def inference_fn(model, loader, device):
    """
    Performs inference with Test-Time Augmentation (TTA).
    TTA Variants: Original, Roll 25%, Roll 50%, Roll 75%.
    """
    model.eval()
    final_preds = []
    rec_ids = []

    # Define roll shifts based on image width
    width = Config.IMG_WIDTH
    shifts = [0, int(width * 0.25), int(width * 0.50), int(width * 0.75)]

    with torch.no_grad():
        for images, _, _, ids in loader:
            images = images.to(device)
            batch_probs = []

            # TTA Loop
            for shift in shifts:
                if shift == 0:
                    inputs = images
                else:
                    # Roll along the width dimension (dim 3: B, C, H, W)
                    inputs = torch.roll(images, shifts=shift, dims=3)

                logits = model(inputs)
                probs = torch.sigmoid(logits)
                batch_probs.append(probs)

            # Average probabilities across TTA variants
            avg_probs = torch.stack(batch_probs).mean(dim=0)

            final_preds.append(avg_probs.cpu().numpy())
            rec_ids.append(ids.numpy())

    final_preds = np.concatenate(final_preds, axis=0)
    rec_ids = np.concatenate(rec_ids, axis=0)

    return final_preds, rec_ids


def save_submission(preds, rec_ids, save_path):
    """
    Formats predictions into the competition submission format.
    Format: Id,Probability
    Where Id = rec_id * 100 + species_id
    """
    submission_ids = []
    submission_probs = []

    num_classes = preds.shape[1]

    for i, rec_id in enumerate(rec_ids):
        for species_idx in range(num_classes):
            # Construct unique ID
            unique_id = int(rec_id * 100 + species_idx)
            prob = preds[i, species_idx]

            submission_ids.append(unique_id)
            submission_probs.append(prob)

    df_sub = pd.DataFrame({"Id": submission_ids, "Probability": submission_probs})

    # Ensure directory exists
    import os

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df_sub.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
