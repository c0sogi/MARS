import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from library.config import Config
from library.model import SaltNet
from library.data import get_test_dataset
from library.utils import set_seed


def load_teacher_ensemble(model_paths, device):
    """
    Loads the ensemble of Specialist Teacher models.

    Args:
        model_paths (list): List of file paths to model checkpoints.
        device (str): Device to load models onto.

    Returns:
        list: List of loaded SaltNet models in evaluation mode.
    """
    models = []
    for path in model_paths:
        if not os.path.exists(path):
            print(f"Warning: Model checkpoint not found at {path}. Skipping.")
            continue

        # Initialize Teacher model (requires depth injection)
        model = SaltNet(mode="teacher")

        # Load weights
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()
        models.append(model)

    if not models:
        raise RuntimeError("No teacher models could be loaded. Check checkpoint paths.")

    return models


def generate_marginalized_labels(
    model_paths, depth_scaler, load_cached_data=True, limit_count=None
):
    """
    Generates soft pseudo-labels for the test set by marginalizing over depth and model uncertainty.

    Strategy:
    1. Load Teacher Ensemble.
    2. Iterate over Test Images.
    3. For each image, scan across Config.DEPTH_SCAN_VALUES.
    4. Average predictions across all models and all depths (Marginalization).
    5. Save and return the dictionary of soft masks.

    Args:
        model_paths (list): Paths to the trained teacher model checkpoints.
        depth_scaler (StandardScaler): Scaler used during training (required for dataset init).
        load_cached_data (bool): If True, attempts to load result from disk.
        limit_count (int, optional): Limit number of images for debugging.

    Returns:
        dict: Dictionary {id: soft_mask_array} where soft_mask_array is (1, 128, 128) float32.
    """
    set_seed(Config.SEED)

    # Define cache path
    cache_path = os.path.join(Config.WORKING_DIR, "pseudo_labels.npy")

    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached pseudo-labels from {cache_path}...")
        try:
            soft_masks = np.load(cache_path, allow_pickle=True).item()
            return soft_masks
        except Exception as e:
            print(f"Failed to load cache: {e}. Regenerating...")

    print("Generating marginalized pseudo-labels...")

    # 2. Setup Data
    test_dataset = get_test_dataset(depth_scaler, load_cached_data=load_cached_data)

    if limit_count is not None:
        indices = range(min(len(test_dataset), limit_count))
        test_dataset = Subset(test_dataset, indices)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Load Models
    device = Config.DEVICE
    models = load_teacher_ensemble(model_paths, device)

    # 4. Inference Loop
    soft_masks = {}
    scan_depths = Config.DEPTH_SCAN_VALUES

    # Total number of predictions per image = Models * Depths * TTA(2)
    normalization_factor = len(models) * len(scan_depths) * 2.0

    with torch.no_grad():
        for images, _, _, ids in tqdm(
            test_loader, desc="Pseudo-Labeling", dynamic_ncols=True
        ):
            images = images.to(device, dtype=torch.float32)
            batch_size = images.size(0)

            # Accumulator for the batch: (B, 1, H, W)
            batch_accum = torch.zeros_like(images[:, 0:1, :, :])

            # Pre-calculate depth tensors for the batch to save time
            # List of (B, 1) tensors
            depth_tensors = [
                torch.full((batch_size, 1), z, device=device, dtype=torch.float32)
                for z in scan_depths
            ]

            # TTA: Horizontal Flip
            images_flipped = torch.flip(images, dims=[3])

            for model in models:
                for z_tensor in depth_tensors:
                    # Forward Pass Original
                    logits = model(images, depth=z_tensor)
                    probs = torch.sigmoid(logits)
                    batch_accum += probs

                    # Forward Pass Flipped
                    logits_flip = model(images_flipped, depth=z_tensor)
                    probs_flip = torch.sigmoid(logits_flip)
                    # Flip back
                    probs_flip_back = torch.flip(probs_flip, dims=[3])
                    batch_accum += probs_flip_back

            # Average
            batch_avg = batch_accum / normalization_factor

            # Store results
            # Move to CPU and numpy
            batch_avg_np = batch_avg.cpu().numpy()

            for i, img_id in enumerate(ids):
                # Store the 128x128 soft mask
                # Shape is (1, 128, 128)
                soft_masks[img_id] = batch_avg_np[i]

    # 5. Save to Cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.save(cache_path, soft_masks)
    print(f"Pseudo-labels saved to {cache_path}")

    return soft_masks
