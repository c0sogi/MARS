import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from library.utils import do_kaggle_metric, rle_encode, unpad_image, set_seed
from library.losses import CombinedLoss

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, optimizer, criterion, device=DEVICE):
    """
    Trains the model for one epoch. Handles both supervised and semi-supervised (distillation) batches.
    """
    model.train()
    running_loss = 0.0
    n_samples = 0

    for batch_idx, data in enumerate(loader):
        # Unpack data
        # Dataset returns: image, depth, mask, flag
        images, depths, masks, flags = data

        images = images.to(device)
        depths = depths.to(device)
        masks = masks.to(device)
        flags = flags.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images, depths)

        # Calculate Loss
        # Split batch into labeled (flag==1) and unlabeled (flag==0)
        labeled_idx = (flags == 1.0).nonzero(as_tuple=True)[0]
        unlabeled_idx = (flags == 0.0).nonzero(as_tuple=True)[0]

        loss = torch.tensor(0.0, device=device)
        count = 0

        # Supervised Loss
        if len(labeled_idx) > 0:
            loss_sup = criterion(
                logits[labeled_idx], masks[labeled_idx], mode="supervised"
            )
            loss += loss_sup * len(labeled_idx)
            count += len(labeled_idx)

        # Distillation Loss (Soft Targets)
        if len(unlabeled_idx) > 0:
            loss_unsup = criterion(
                logits[unlabeled_idx], masks[unlabeled_idx], mode="distillation"
            )
            loss += loss_unsup * len(unlabeled_idx)
            count += len(unlabeled_idx)

        if count > 0:
            loss = loss / count
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * count
            n_samples += count

    return running_loss / n_samples if n_samples > 0 else 0.0


def validate(model, loader, criterion, device=DEVICE):
    """
    Evaluates the model on the validation set.
    Returns average loss, default mAP (threshold 0.5), and raw predictions/targets for threshold search.
    """
    model.eval()
    running_loss = 0.0
    n_samples = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data in loader:
            # Validation dataset returns: image, depth, mask, flag
            images, depths, masks, _ = data

            images = images.to(device)
            depths = depths.to(device)
            masks = masks.to(device)

            logits = model(images, depths)

            # Loss calculation (Supervised)
            loss = criterion(logits, masks, mode="supervised")

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

            # Store probabilities and targets for metric calculation
            probs = torch.sigmoid(logits).cpu().numpy()
            # Remove channel dim: (B, 1, H, W) -> (B, H, W)
            if probs.ndim == 4:
                probs = probs.squeeze(1)

            targets = masks.cpu().numpy()
            if targets.ndim == 4:
                targets = targets.squeeze(1)

            all_preds.append(probs)
            all_targets.append(targets)

    avg_loss = running_loss / n_samples if n_samples > 0 else 0.0

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate default mAP (threshold 0.5)
    # Note: do_kaggle_metric expects binary predictions if threshold is provided,
    # or it binarizes internally. We pass probs and let it handle or binarize here.
    # The utils function binarizes if dtype is not uint8/bool.
    default_map = do_kaggle_metric(all_preds, all_targets, threshold=0.5)

    return avg_loss, default_map, all_preds, all_targets


def threshold_search(preds, targets):
    """
    Finds the optimal binarization threshold that maximizes the competition mAP.
    """
    best_threshold = 0.5
    best_score = -1.0

    # Search range: 0.3 to 0.7 usually covers the optimal point for IoU metrics
    thresholds = np.linspace(0.3, 0.7, 41)

    for t in thresholds:
        score = do_kaggle_metric(preds, targets, threshold=t)
        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold, best_score


def generate_soft_targets(model, loader, device=DEVICE):
    """
    Generates soft probability maps for the test set using the Teacher model.
    Applies TTA (Horizontal Flip) and forces depth to 0.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for data in loader:
            # Test dataset returns: image, depth
            images, _ = data
            images = images.to(device)

            # Force depth to 0 for robustness
            depths = torch.zeros(
                (images.size(0), 1), device=device, dtype=torch.float32
            )

            # TTA: Original
            logits_orig = model(images, depths)
            probs_orig = torch.sigmoid(logits_orig)

            # TTA: Flip
            images_flipped = torch.flip(images, dims=[3])
            logits_flip = model(images_flipped, depths)
            probs_flip = torch.sigmoid(logits_flip)
            probs_flip = torch.flip(probs_flip, dims=[3])

            # Average
            probs = (probs_orig + probs_flip) / 2.0

            # Squeeze channel
            if probs.ndim == 4:
                probs = probs.squeeze(1)

            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


def inference(model, loader, threshold, output_path="submission.csv", device=DEVICE):
    """
    Runs inference on the test set, applies TTA and threshold, unpads images,
    encodes to RLE, and saves to CSV.
    """
    model.eval()
    rle_masks = []
    ids = []

    # We need IDs. The loader from dataset.py doesn't return IDs in __getitem__.
    # However, the dataset object has the ids attribute.
    # We will assume the loader iterates sequentially matching dataset.ids.
    dataset_ids = (
        loader.dataset.is_labeled
    )  # This is actually flags in the dataset class...
    # Wait, the dataset class in `library/dataset.py` stores `self.ids` in `prepare_data` cache,
    # but `SaltDataset` class doesn't store IDs directly in `__init__` unless we modify it.
    # Looking at `library/dataset.py`: `prepare_data` saves `test_ids.npy`.
    # We should load test_ids directly from the cache or metadata to ensure alignment.

    # Load test IDs
    test_ids = np.load("./working/idea_17/test_ids.npy", allow_pickle=True)

    current_idx = 0

    with torch.no_grad():
        for data in loader:
            # Test dataset returns: image, depth
            images, _ = data
            images = images.to(device)
            batch_size = images.size(0)

            # Force depth to 0
            depths = torch.zeros((batch_size, 1), device=device, dtype=torch.float32)

            # TTA
            logits_orig = model(images, depths)
            probs_orig = torch.sigmoid(logits_orig)

            images_flipped = torch.flip(images, dims=[3])
            logits_flip = model(images_flipped, depths)
            probs_flip = torch.sigmoid(logits_flip)
            probs_flip = torch.flip(probs_flip, dims=[3])

            probs = (probs_orig + probs_flip) / 2.0

            # Binarize
            preds = (probs > threshold).float()

            # Post-process
            preds_np = preds.cpu().numpy()  # (B, 1, 128, 128) or (B, 128, 128)
            if preds_np.ndim == 4:
                preds_np = preds_np.squeeze(1)

            for i in range(batch_size):
                # Unpad 128x128 -> 101x101
                mask_128 = preds_np[i]
                mask_101 = unpad_image(mask_128, original_size=(101, 101))

                # Binarize again just to be safe (unpad is slicing, so values are preserved)
                mask_101 = (mask_101 > 0.5).astype(np.uint8)

                # RLE Encode
                rle = rle_encode(mask_101)
                rle_masks.append(rle)
                ids.append(test_ids[current_idx])
                current_idx += 1

    # Create DataFrame
    df = pd.DataFrame({"id": ids, "rle_mask": rle_masks})

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
