import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import SaltDataset, set_seed
from library.model import HighCapacityUNet
from library.utils import rle_encode


def center_crop(tensor, target_h, target_w):
    """
    Center crops a tensor to the target spatial dimensions.
    Assumes tensor shape is (..., H, W).
    """
    h, w = tensor.shape[-2:]
    diff_h = (h - target_h) // 2
    diff_w = (w - target_w) // 2
    return tensor[..., diff_h : diff_h + target_h, diff_w : diff_w + target_w]


def load_checkpoint(model, path, device):
    """
    Loads model weights from a checkpoint path.
    """
    print(f"Loading checkpoint: {path}")
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_ensemble(debug=False, limit=None):
    """
    Generates predictions using Snapshot Ensembling (Cycle 2 + Cycle 3)
    with Test-Time Augmentation (Horizontal Flip).

    Args:
        debug (bool): If True, runs on a subset of data.
        limit (int): Number of samples to process if debug is True.
    """
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Inference Device: {device}")

    # 2. Data Loading
    print("Initializing Test Dataset...")
    test_dataset = SaltDataset(
        mode="test", load_cached_data=True, limit=limit if debug else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Identify Checkpoints
    # We aim to ensemble Cycle 2 and Cycle 3 best models.
    # Fallback to best_model.pth if specific cycles are missing (e.g. short training).
    checkpoint_paths = []

    c2_path = os.path.join(Config.CHECKPOINT_DIR, "best_cycle_2.pth")
    c3_path = os.path.join(Config.CHECKPOINT_DIR, "best_cycle_3.pth")
    best_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if os.path.exists(c2_path):
        checkpoint_paths.append(c2_path)
    if os.path.exists(c3_path):
        checkpoint_paths.append(c3_path)

    if not checkpoint_paths:
        if os.path.exists(best_path):
            print(
                "Warning: Cycle checkpoints not found. Falling back to best_model.pth"
            )
            checkpoint_paths.append(best_path)
        else:
            raise FileNotFoundError("No checkpoints found in " + Config.CHECKPOINT_DIR)

    print(
        f"Ensembling {len(checkpoint_paths)} models: {[os.path.basename(p) for p in checkpoint_paths]}"
    )

    # 4. Inference Loop
    results = []

    # We load one model structure and reload weights to save memory,
    # or load multiple if memory permits. Given A100 40GB, we can load multiple.
    # However, sequentially loading weights is safer and generic.

    # Pre-instantiate model structure
    model = HighCapacityUNet().to(device)

    # To optimize, we will iterate through the dataset once, and inside the loop
    # we can't easily swap weights.
    # Better approach for Ensemble: Load all models into a list if memory allows.
    # HighCapacityUNet is heavy but 40GB VRAM is plenty for 2-3 copies.

    models = []
    for path in checkpoint_paths:
        m = HighCapacityUNet().to(device)
        m = load_checkpoint(m, path, device)
        models.append(m)

    print("Starting Inference...")

    with torch.no_grad():
        for images, _, depths, ids in test_loader:
            images = images.to(device)
            depths = depths.to(device)

            # TTA: Horizontal Flip
            images_flip = torch.flip(images, dims=[3])

            # Accumulate probabilities
            ensemble_probs = torch.zeros_like(images)

            count = 0
            for m in models:
                # Original
                logits = m(images, depths)
                probs = torch.sigmoid(logits)
                ensemble_probs += probs
                count += 1

                # Flipped
                logits_flip = m(images_flip, depths)
                probs_flip = torch.sigmoid(logits_flip)
                # Flip back
                probs_flip_back = torch.flip(probs_flip, dims=[3])
                ensemble_probs += probs_flip_back
                count += 1

            # Average
            avg_probs = ensemble_probs / count

            # Crop to original size (101x101)
            # Dataset pads to 128x128. Center crop removes padding.
            cropped_probs = center_crop(avg_probs, Config.ORIG_SIZE, Config.ORIG_SIZE)

            # Threshold
            pred_masks = (cropped_probs > 0.5).float().cpu().numpy()

            # Encode
            # pred_masks shape: (B, 1, 101, 101)
            for i, img_id in enumerate(ids):
                mask = pred_masks[i, 0, :, :]  # (101, 101)
                rle = rle_encode(mask)
                results.append({"id": img_id, "rle_mask": rle})

    # 5. Save Submission
    df_sub = pd.DataFrame(results)

    # Ensure column order
    df_sub = df_sub[["id", "rle_mask"]]

    save_path = Config.SUBMISSION_FILE
    print(f"Saving submission to {save_path}")
    df_sub.to_csv(save_path, index=False)
    print("Inference completed successfully.")
