import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from library.config import CFG
from library.utils import rle_encoding, seed_everything
from library.data import InkDataset
from library.model import WideContextSegFormer


def get_tta_predictions(model, images):
    """
    Applies Test Time Augmentation (TTA) by averaging predictions from:
    1. Original
    2. Horizontal Flip
    3. Vertical Flip
    4. Rotate 90 degrees
    """
    # 1. Original
    logits = model(images)
    probs = torch.sigmoid(logits)

    # 2. Horizontal Flip
    images_h = torch.flip(images, dims=[3])
    logits_h = model(images_h)
    probs_h = torch.flip(torch.sigmoid(logits_h), dims=[3])

    # 3. Vertical Flip
    images_v = torch.flip(images, dims=[2])
    logits_v = model(images_v)
    probs_v = torch.flip(torch.sigmoid(logits_v), dims=[2])

    # 4. Rotate 90 (k=1, dims=(2, 3))
    images_r = torch.rot90(images, k=1, dims=[2, 3])
    logits_r = model(images_r)
    probs_r = torch.rot90(torch.sigmoid(logits_r), k=-1, dims=[2, 3])

    # Average
    avg_probs = (probs + probs_h + probs_v + probs_r) / 4.0
    return avg_probs


def predict_fragment(fragment_id, metadata_df, model, device, debug_max_batches=None):
    """
    Performs inference on a single fragment.

    Args:
        fragment_id (str): ID of the fragment to predict.
        metadata_df (pd.DataFrame): DataFrame containing test metadata.
        model (nn.Module): Loaded model.
        device (torch.device): Device to run inference on.
        debug_max_batches (int, optional): Limit number of batches for debugging.

    Returns:
        np.ndarray: Binary mask of predictions (0 or 1).
    """
    # Filter metadata for this specific fragment
    frag_df = metadata_df[metadata_df["fragment_id"] == fragment_id].copy()

    # Initialize Dataset and DataLoader
    # We use load_cached_data=True to utilize any pre-processed volumes
    dataset = InkDataset(frag_df, mode="test", load_cached_data=True)
    dataloader = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Retrieve full mask shape to initialize the probability map
    # InkDataset loads the full fragment data into memory upon init
    full_mask = dataset.fragments_data[fragment_id]["mask"]
    full_h, full_w = full_mask.shape

    prob_map = np.zeros((full_h, full_w), dtype=np.float32)

    model.eval()

    with torch.no_grad():
        for batch_idx, (images, _, indices) in enumerate(dataloader):
            if debug_max_batches is not None and batch_idx >= debug_max_batches:
                break

            images = images.to(device)

            # Get predictions with TTA
            batch_probs = get_tta_predictions(model, images)
            batch_probs = batch_probs.cpu().numpy()

            # Stitch patches back into the full probability map
            for i, idx in enumerate(indices):
                # Retrieve patch coordinates from the dataset
                sample = dataset.samples[idx]
                x = sample["x"]
                y = sample["y"]
                w = sample["width"]
                h = sample["height"]

                # Calculate valid dimensions (handling padding at edges)
                # The model outputs fixed size (e.g., 512x512), but if the patch
                # extended beyond the image boundary, it was padded.
                # We only want to paste the valid region.
                valid_h = min(y + h, full_h) - y
                valid_w = min(x + w, full_w) - x

                # Extract valid prediction area
                # batch_probs shape: (B, 1, H_patch, W_patch)
                pred_patch = batch_probs[i, 0, :valid_h, :valid_w]

                # Place into global map
                prob_map[y : y + valid_h, x : x + valid_w] = pred_patch

    # Apply the fragment mask to zero out invalid areas (outside the papyrus)
    prob_map = prob_map * (full_mask > 0)

    # Threshold to binary
    binary_pred = (prob_map > CFG.threshold).astype(np.uint8)

    return binary_pred


def inference(
    model_path=CFG.model_path, submission_path=CFG.submission_path, debug_samples=None
):
    """
    Main inference pipeline. Generates predictions for all test fragments
    and saves the submission file.

    Args:
        model_path (str): Path to the trained model checkpoint.
        submission_path (str): Path to save the submission CSV.
        debug_samples (int, optional): If set, limits inference to a subset of batches.
    """
    seed_everything(CFG.seed)

    # Check for test metadata
    if not os.path.exists(CFG.test_metadata_path):
        print(
            f"Test metadata not found at {CFG.test_metadata_path}. Skipping inference."
        )
        return

    test_df = pd.read_csv(CFG.test_metadata_path)
    fragment_ids = test_df["fragment_id"].unique()

    print(f"Found {len(fragment_ids)} test fragments.")

    # Load Model
    print(f"Loading model from {model_path}...")
    model = WideContextSegFormer(CFG)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=CFG.device)
        # Handle case where checkpoint saves 'model_state_dict' or just state_dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(
            f"Warning: Model checkpoint not found at {model_path}. Using random weights."
        )

    model.to(CFG.device)

    results = []

    for fid in fragment_ids:
        print(f"Processing fragment {fid}...")

        # Run inference
        binary_mask = predict_fragment(
            fid, test_df, model, CFG.device, debug_max_batches=debug_samples
        )

        # Encode
        rle = rle_encoding(binary_mask)
        results.append({"Id": fid, "Predicted": rle})

    # Save Submission
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
