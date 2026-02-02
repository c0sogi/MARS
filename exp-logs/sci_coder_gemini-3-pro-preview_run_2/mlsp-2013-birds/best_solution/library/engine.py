import torch
import numpy as np
from library.config import CFG
from library.utils import calculate_metric


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Standard training loop with Mixup augmentation.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)

        # Apply Mixup
        if CFG.mixup_alpha > 0:
            lam = np.random.beta(CFG.mixup_alpha, CFG.mixup_alpha)
            index = torch.randperm(batch_size).to(device)

            mixed_images = lam * images + (1 - lam) * images[index]
            mixed_targets = lam * targets + (1 - lam) * targets[index]

            logits = model(mixed_images)
            loss = criterion(logits, mixed_targets)
        else:
            logits = model(images)
            loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def train_distill_one_epoch(student_model, loader, optimizer, criterion, device, epoch):
    """
    Training loop for Knowledge Distillation.

    Expects the loader to yield 'packed' labels where:
    - Columns [0 : num_classes] are Ground Truth (Binary)
    - Columns [num_classes : 2*num_classes] are Teacher Soft Probabilities
    """
    student_model.train()
    running_loss = 0.0
    dataset_size = 0

    num_classes = CFG.num_classes

    for batch_idx, (images, packed_labels) in enumerate(loader):
        images = images.to(device)
        packed_labels = packed_labels.to(device)
        batch_size = images.size(0)

        # Unpack labels
        # Ensure we slice correctly based on configuration
        targets = packed_labels[:, :num_classes]
        teacher_probs = packed_labels[:, num_classes:]

        # Apply Mixup to Images, Targets, AND Teacher Probs
        if CFG.mixup_alpha > 0:
            lam = np.random.beta(CFG.mixup_alpha, CFG.mixup_alpha)
            index = torch.randperm(batch_size).to(device)

            mixed_images = lam * images + (1 - lam) * images[index]
            mixed_targets = lam * targets + (1 - lam) * targets[index]
            mixed_teacher_probs = lam * teacher_probs + (1 - lam) * teacher_probs[index]

            logits = student_model(mixed_images)
            # WeightedDistillationLoss takes (student_logits, targets, teacher_probs)
            loss = criterion(logits, mixed_targets, mixed_teacher_probs)
        else:
            logits = student_model(images)
            loss = criterion(logits, targets, teacher_probs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def valid_one_epoch(model, loader, criterion, device):
    """
    Validation loop. Computes Loss and AUC.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)

            logits = model(images)
            loss = criterion(logits, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Store predictions for AUC calculation
            probs = torch.sigmoid(logits)
            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate all batches
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # Calculate Macro-AUC
    epoch_auc = calculate_metric(all_targets, all_preds)

    return epoch_loss, epoch_auc


def predict_with_tta(model, loader, device):
    """
    Generates predictions using Cyclic Test-Time Augmentation (TTA).
    Variants: Original, Roll 25%, Roll 50%, Roll 75% along the time axis.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            # images shape: (B, C, H, W)
            W = images.shape[3]

            # Define shifts: 0, 1/4, 1/2, 3/4 of width
            shifts = [0, W // 4, W // 2, (3 * W) // 4]

            batch_probs_sum = None

            for s in shifts:
                if s == 0:
                    aug_images = images
                else:
                    # Cyclic roll along width (dim 3)
                    aug_images = torch.roll(images, shifts=s, dims=3)

                logits = model(aug_images)
                probs = torch.sigmoid(logits)

                if batch_probs_sum is None:
                    batch_probs_sum = probs
                else:
                    batch_probs_sum += probs

            # Average probabilities across TTA variants
            batch_avg_probs = batch_probs_sum / len(shifts)
            all_preds.append(batch_avg_probs.cpu().numpy())

    return np.concatenate(all_preds, axis=0)
