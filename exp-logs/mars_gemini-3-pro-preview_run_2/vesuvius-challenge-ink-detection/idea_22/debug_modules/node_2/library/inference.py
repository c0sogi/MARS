import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, rle_encoding
from library.dataset import InkDataset
from library.model import SpecialistModel


def predict_slab(model, loader, shape, device):
    """
    Generates a probability map for a specific specialist model over the entire fragment.
    Stitches tiles together using averaging for overlaps.

    Args:
        model: Loaded PyTorch SpecialistModel.
        loader: DataLoader containing the tiled test data.
        shape: Tuple (H, W) of the full fragment.
        device: Torch device.

    Returns:
        Numpy array (H, W) with pixel probabilities.
    """
    # Accumulators for stitching
    prob_map = np.zeros(shape, dtype=np.float32)
    count_map = np.zeros(shape, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            # Coordinates of the top-left corner of the tile
            xs = batch["x"].numpy()
            ys = batch["y"].numpy()

            # Predict
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()  # (B, 1, H, W)

            # Stitch into full map
            for i in range(len(images)):
                p = probs[i, 0]  # (H_tile, W_tile)
                x, y = xs[i], ys[i]
                h_tile, w_tile = p.shape

                # Determine placement coordinates in the full image
                y_start, x_start = y, x
                y_end = min(y + h_tile, shape[0])
                x_end = min(x + w_tile, shape[1])

                # Determine valid region within the tile (handling padding at bottom/right)
                # The tile is 512x512. If it goes off the image, InkDataset padded it.
                # We only want the part that corresponds to the image.
                h_valid = y_end - y_start
                w_valid = x_end - x_start

                if h_valid <= 0 or w_valid <= 0:
                    continue

                # Add to accumulators
                prob_map[y_start:y_end, x_start:x_end] += p[:h_valid, :w_valid]
                count_map[y_start:y_end, x_start:x_end] += 1.0

    # Normalize by count to handle overlaps
    # Avoid division by zero
    mask = count_map > 0
    prob_map[mask] /= count_map[mask]

    return prob_map


def fuse_predictions(predictions):
    """
    Performs Max-Fusion on a list of probability maps.

    Args:
        predictions: List of numpy arrays (H, W).

    Returns:
        Numpy array (H, W) representing the fused probability map.
    """
    if not predictions:
        return None

    # Start with the first map
    fused_map = predictions[0].copy()

    # Maximize with subsequent maps
    for p in predictions[1:]:
        fused_map = np.maximum(fused_map, p)

    return fused_map


def run_inference(load_cached_data=True):
    """
    Main inference routine.
    Generates predictions for all fragments in the test set using the MDSE ensemble.
    Saves the submission file.

    Args:
        load_cached_data: Whether to use cached .npy volume slabs.
    """
    set_seed(Config.SEED)

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH) or ".", exist_ok=True)

    # Load Test Metadata
    if not os.path.exists(Config.TEST_METADATA):
        print(f"Error: Test metadata not found at {Config.TEST_METADATA}")
        return

    test_df = pd.read_csv(Config.TEST_METADATA)
    fragment_ids = test_df["fragment_id"].unique()

    submission_rows = []

    print(f"Starting inference on {len(fragment_ids)} fragments.")

    for fid in fragment_ids:
        print(f"Processing Fragment {fid}...")

        # 1. Get Fragment Info & Shape
        # We need the mask to determine the full shape and to mask out invalid areas later.
        frag_row = test_df[test_df["fragment_id"] == fid].iloc[0]
        mask_path = os.path.join(Config.INPUT_DIR, frag_row["mask_path"])

        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            print(f"  Error: Mask not found for fragment {fid}. Skipping.")
            continue

        full_shape = mask_img.shape  # (H, W)

        # 2. Generate Predictions from Each Specialist
        specialist_maps = []

        for spec_config in Config.SPECIALISTS:
            spec_name = spec_config["name"]
            z_start = spec_config["z_start"]
            z_end = spec_config["z_end"]
            ckpt_path = spec_config["checkpoint_path"]

            print(f"  Running Specialist: {spec_name} (Z: {z_start}-{z_end})")

            # Check if model exists
            if not os.path.exists(ckpt_path):
                print(f"    Warning: Checkpoint {ckpt_path} not found. Skipping.")
                continue

            # Initialize Model
            model = SpecialistModel(model_name=Config.BACKBONE)
            state_dict = torch.load(ckpt_path, map_location=Config.DEVICE)
            model.load_state_dict(state_dict)
            model.to(Config.DEVICE)

            # Prepare Data
            # Filter metadata for just this fragment
            frag_meta = test_df[test_df["fragment_id"] == fid].copy()

            dataset = InkDataset(
                metadata=frag_meta,
                z_start=z_start,
                z_end=z_end,
                mode="test",
                load_cached_data=load_cached_data,
            )

            loader = DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Predict
            prob_map = predict_slab(model, loader, full_shape, Config.DEVICE)
            specialist_maps.append(prob_map)

            # Clean up to save memory
            del model
            torch.cuda.empty_cache()

        # 3. Fuse Predictions (Max-Fusion)
        if not specialist_maps:
            print(
                f"  Warning: No predictions generated for fragment {fid}. Outputting zeros."
            )
            final_prob = np.zeros(full_shape, dtype=np.float32)
        else:
            final_prob = fuse_predictions(specialist_maps)

        # 4. Post-Processing
        # Mask out invalid areas (outside the papyrus fragment)
        valid_mask = mask_img > 0
        final_prob[~valid_mask] = 0

        # Binarize
        binary_pred = (final_prob > 0.5).astype(np.uint8)

        # 5. Encode
        rle = rle_encoding(binary_pred)
        submission_rows.append({"Id": fid, "Predicted": rle})

    # Save Submission
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Inference complete. Submission saved to {Config.SUBMISSION_PATH}")
