import torch
import numpy as np
from library.models import reparameterize_model


def apply_tta(model, images, fsizes, device):
    """
    Applies 4-view Test Time Augmentation (Original, H-Flip, V-Flip, Rot180).

    Args:
        model: The neural network model.
        images: Batch of images (B, C, H, W).
        fsizes: Batch of file size metadata (B,).
        device: Torch device.

    Returns:
        avg_probs: Averaged probabilities across all views (B, 1).
    """
    # Ensure inputs are on the correct device
    images = images.to(device)
    fsizes = fsizes.to(device)

    # List to store probabilities from different views
    probs_list = []

    # Define views: Original, H-Flip, V-Flip, 180-Rotate
    # Images are (B, C, H, W), so H is dim 2, W is dim 3
    views = [
        images,  # Original
        torch.flip(images, dims=[3]),  # Horizontal Flip
        torch.flip(images, dims=[2]),  # Vertical Flip
        torch.flip(images, dims=[2, 3]),  # Rotate 180
    ]

    for view in views:
        # Forward pass
        # Model returns dict with 'logits'
        outputs = model(view, fsizes)
        logits = outputs["logits"]

        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        probs_list.append(probs)

    # Stack results (4, B, 1) and calculate mean across views (dim 0)
    probs_stack = torch.stack(probs_list)
    avg_probs = torch.mean(probs_stack, dim=0)

    return avg_probs


def predict_loader(model, loader, device):
    """
    Generates predictions for a full DataLoader using TTA and RepVGG fusion.

    Args:
        model: PyTorch model instance.
        loader: DataLoader yielding (images, labels, fsizes, ids).
        device: Torch device.

    Returns:
        preds: Numpy array of predicted probabilities (N,).
        ids: Numpy array of corresponding image IDs (N,).
    """
    # Set model to evaluation mode
    model.eval()

    # Structural Re-parameterization:
    # Fuse Conv+BN branches in RepVGG blocks into a single Conv for faster inference.
    # This modifies the model in-place.
    reparameterize_model(model)

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, _, fsizes, img_ids in loader:
            # Generate predictions with Test Time Augmentation
            batch_probs = apply_tta(model, images, fsizes, device)

            # Move to CPU and store
            all_preds.append(batch_probs.cpu().numpy())

            # Handle IDs (which might be tuples of strings or tensors)
            if isinstance(img_ids, torch.Tensor):
                all_ids.append(img_ids.cpu().numpy())
            else:
                all_ids.append(img_ids)

    # Aggregate results
    if len(all_preds) > 0:
        # Concatenate and flatten to 1D array
        preds = np.concatenate(all_preds).flatten()
        ids = np.concatenate(all_ids)
    else:
        preds = np.array([])
        ids = np.array([])

    return preds, ids
