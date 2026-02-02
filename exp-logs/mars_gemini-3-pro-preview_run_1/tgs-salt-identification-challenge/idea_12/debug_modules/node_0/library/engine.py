import torch
import torch.nn.functional as F
import numpy as np
from library.config import Config
from library.utils import AverageMeter, compute_map_batch


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Executes one epoch of training.

    Args:
        model (nn.Module): The segmentation model.
        loader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function (ConsistentCompoundLoss).
        optimizer (Optimizer): Optimizer.
        device (torch.device): Compute device.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (images, masks, depths, _) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        depths = depths.to(device)

        optimizer.zero_grad()

        # Forward pass
        # Model may return a list [logits, aux1, aux2] if deep supervision is active
        outputs = model(images, depths)

        if isinstance(outputs, list):
            # Primary output
            logits = outputs[0]
            loss = criterion(logits, masks)

            # Auxiliary outputs (Deep Supervision)
            # We weight auxiliary losses by 0.5
            for aux in outputs[1:]:
                # Interpolate aux output to match mask spatial dimensions (128x128)
                aux_upsampled = F.interpolate(
                    aux, size=masks.shape[2:], mode="bilinear", align_corners=True
                )
                loss += 0.5 * criterion(aux_upsampled, masks)
        else:
            # Standard output
            logits = outputs
            loss = criterion(logits, masks)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Computes mAP on the original 101x101 image size by center-cropping
    predictions and masks to remove padding artifacts.

    Args:
        model (nn.Module): The segmentation model.
        loader (DataLoader): Validation data loader.
        device (torch.device): Compute device.

    Returns:
        float: Mean Average Precision (mAP) over the validation set.
    """
    model.eval()
    map_score = AverageMeter()

    # Target dimensions for metric calculation (original image size)
    orig_h, orig_w = Config.ORIG_IMG_SIZE, Config.ORIG_IMG_SIZE

    with torch.no_grad():
        for images, masks, depths, _ in loader:
            images = images.to(device)
            depths = depths.to(device)
            # Note: masks are (B, 1, 128, 128) due to dataset padding

            outputs = model(images, depths)

            # Handle potential list output (though usually tensor in eval mode)
            if isinstance(outputs, list):
                outputs = outputs[0]

            probs = torch.sigmoid(outputs)

            # Center Crop Predictions and Masks back to 101x101
            # The dataset pads 101 -> 128 symmetrically/reflected.
            # We must evaluate on the central 101x101 region.
            h, w = probs.shape[2], probs.shape[3]
            start_h = (h - orig_h) // 2
            start_w = (w - orig_w) // 2

            # Crop predictions (GPU)
            probs_cropped = probs[
                :, :, start_h : start_h + orig_h, start_w : start_w + orig_w
            ]

            # Crop masks (CPU)
            # We move masks to CPU first if they are on GPU (though usually loaded on CPU)
            masks_cpu = masks.cpu()
            masks_cropped = masks_cpu[
                :, :, start_h : start_h + orig_h, start_w : start_w + orig_w
            ]

            # Binarize predictions
            preds = (probs_cropped > 0.5).float().cpu().numpy()
            targets = masks_cropped.numpy()

            # Compute mAP
            # Squeeze channel dim: (B, 1, H, W) -> (B, H, W)
            # compute_map_batch expects flattened arrays internally but takes (B, H, W)
            batch_map = compute_map_batch(preds.squeeze(1), targets.squeeze(1))
            map_score.update(batch_map, images.size(0))

    # Print full precision metric as requested
    print(f"Validation mAP: {map_score.avg}")

    return map_score.avg
