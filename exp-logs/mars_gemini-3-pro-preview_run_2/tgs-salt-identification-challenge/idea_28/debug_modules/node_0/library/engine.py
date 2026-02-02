import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from library.config import Config
from library.utils import rle_encode, get_score
from library.losses import MultiTaskLoss


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: Training dataloader.
        criterion: The MultiTaskLoss function.
        optimizer: The optimizer.
        device: 'cuda' or 'cpu'.
        epoch: Current epoch number (for logging).

    Returns:
        dict: Average losses for the epoch.
    """
    model.train()

    running_loss = 0.0
    running_bce = 0.0
    running_lovasz = 0.0
    running_depth = 0.0

    dataset_size = 0

    for batch_idx, batch in enumerate(dataloader):
        # Unpack batch: image, mask, depth, id
        if len(batch) == 4:
            images, masks, depths, ids = batch
        else:
            continue

        images = images.to(device)
        masks = masks.to(device)
        depths = depths.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        # Forward pass
        mask_logits, depth_pred = model(images)

        # ---------------------------------------------------------------------
        # CRITICAL: Runtime Assertion for Depth Head Connection
        # ---------------------------------------------------------------------
        # We must verify that the auxiliary depth head is connected to the
        # computational graph to prevent the "silent disconnection" bug.
        # Since we cannot access the internal 'loss_depth' tensor from the
        # criterion (which returns it as a float in the metrics dict),
        # we verify that the prediction tensor itself requires gradients.
        if not depth_pred.requires_grad:
            raise RuntimeError(
                "Critical Error: Depth prediction head is disconnected from the "
                "computational graph! 'depth_pred.requires_grad' is False."
            )

        # Compute Loss
        loss, metrics = criterion(mask_logits, depth_pred, masks, depths)

        # Verify total loss requires grad
        if not loss.requires_grad:
            raise RuntimeError("Critical Error: Total loss does not require gradients.")

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        running_loss += loss.item() * batch_size
        running_bce += metrics["loss_bce"] * batch_size
        running_lovasz += metrics["loss_lovasz"] * batch_size
        running_depth += metrics["loss_depth"] * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    epoch_bce = running_bce / dataset_size
    epoch_lovasz = running_lovasz / dataset_size
    epoch_depth = running_depth / dataset_size

    print(
        f"Train Epoch {epoch}: Loss={epoch_loss:.5f} (BCE={epoch_bce:.5f}, Lovasz={epoch_lovasz:.5f}, Depth={epoch_depth:.5f})"
    )

    return {
        "loss": epoch_loss,
        "bce": epoch_bce,
        "lovasz": epoch_lovasz,
        "depth": epoch_depth,
    }


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: Validation dataloader.
        criterion: The MultiTaskLoss function.
        device: 'cuda' or 'cpu'.

    Returns:
        dict: Validation metrics (loss, map).
    """
    model.eval()

    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                images, masks, depths, ids = batch
            else:
                continue

            images = images.to(device)
            masks = masks.to(device)
            depths = depths.to(device)
            batch_size = images.size(0)

            mask_logits, depth_pred = model(images)

            loss, metrics = criterion(mask_logits, depth_pred, masks, depths)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for probabilities
            preds = torch.sigmoid(mask_logits)

            # Store for mAP calculation (move to CPU)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate predictions
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Remove channel dimension if present: (N, 1, H, W) -> (N, H, W)
    if all_preds.ndim == 4:
        all_preds = all_preds[:, 0, :, :]
    if all_targets.ndim == 4:
        all_targets = all_targets[:, 0, :, :]

    # Crop back to original 101x101 size for accurate metric calculation
    # The model outputs 128x128 (padded).
    start = (Config.IMG_SIZE - Config.ORIG_SIZE) // 2
    end = start + Config.ORIG_SIZE

    preds_crop = all_preds[:, start:end, start:end]
    targets_crop = all_targets[:, start:end, start:end]

    # Calculate mAP at threshold 0.5
    bin_preds = (preds_crop > 0.5).astype(np.uint8)
    map_score = get_score(bin_preds, targets_crop)

    print(f"Val: Loss={epoch_loss:.5f}, mAP={map_score:.5f}")

    return {"val_loss": epoch_loss, "map": map_score}


def predict_tta(model, dataloader, device):
    """
    Generates predictions using Test-Time Augmentation (Horizontal Flip).

    Args:
        model: The PyTorch model.
        dataloader: Test dataloader.
        device: 'cuda' or 'cpu'.

    Returns:
        tuple: (predictions_array, ids_list)
               predictions_array is (N, 1, 101, 101) probabilities.
    """
    model.eval()

    all_preds = []
    all_ids = []

    # Crop indices
    start = (Config.IMG_SIZE - Config.ORIG_SIZE) // 2
    end = start + Config.ORIG_SIZE

    with torch.no_grad():
        for batch in dataloader:
            # Test loader returns: image, depth, id
            images, depths, ids = batch
            images = images.to(device)

            # 1. Original Prediction
            logits, _ = model(images)
            probs = torch.sigmoid(logits)

            # 2. Horizontal Flip TTA
            images_flip = torch.flip(images, dims=[3])  # Flip width (dim 3)
            logits_flip, _ = model(images_flip)
            probs_flip = torch.sigmoid(logits_flip)
            probs_flip = torch.flip(probs_flip, dims=[3])  # Flip back

            # Average predictions
            avg_probs = (probs + probs_flip) / 2.0

            # Crop to original size 101x101
            avg_probs_crop = avg_probs[:, :, start:end, start:end]

            all_preds.append(avg_probs_crop.cpu().numpy())
            all_ids.extend(ids)

    return np.concatenate(all_preds, axis=0), all_ids


def generate_submission(
    model,
    dataloader,
    device,
    output_path=os.path.join(Config.SUBMISSION_DIR, "submission.csv"),
    threshold=0.5,
):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model: The trained model.
        dataloader: Test dataloader.
        device: 'cuda' or 'cpu'.
        output_path: Path to save the submission CSV.
        threshold: Threshold for binarizing probabilities.
    """
    print("Generating submission with TTA...")
    preds, ids = predict_tta(model, dataloader, device)

    # Binarize
    # preds shape: (N, 1, 101, 101)
    if preds.ndim == 4:
        preds = preds[:, 0, :, :]

    binary_preds = (preds > threshold).astype(np.uint8)

    # Encode to RLE
    rle_masks = []
    for i in range(len(ids)):
        rle = rle_encode(binary_preds[i])
        rle_masks.append(rle)

    # Save
    df = pd.DataFrame({"id": ids, "rle_mask": rle_masks})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
