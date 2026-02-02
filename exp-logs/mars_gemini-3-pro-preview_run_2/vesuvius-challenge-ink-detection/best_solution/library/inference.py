import os
import torch
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import seed_everything, rle_encoding
from library.model import build_model
from library.dataset import InkDataset


def inference(
    threshold=Config.THRESHOLD,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
    use_tta=Config.USE_TTA,
):
    """
    Runs the inference pipeline: loads data, predicts with TTA, stitches tiles,
    and generates the RLE-encoded submission file.

    Args:
        threshold (float): Probability threshold for binary classification.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of worker processes for data loading.
        debug (bool): If True, runs on a subset of data.
        use_tta (bool): If True, applies Test Time Augmentation.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Load Metadata
    if not os.path.exists(Config.TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"Test metadata not found at {Config.TEST_METADATA_PATH}"
        )

    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    if debug:
        test_df = test_df.head(1)
        print("Debug mode: Running inference on 1 fragment.")

    # 3. Prepare Dataset and Loader
    # load_cached_data=False ensures we process the actual input volumes
    # present at runtime (handling hidden test set substitution).
    dataset = InkDataset(test_df, mode="test", load_cached_data=False)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 4. Load Model
    model = build_model()
    if os.path.exists(Config.MODEL_SAVE_PATH):
        checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded model weights from {Config.MODEL_SAVE_PATH}")
    else:
        print(
            f"Warning: Model weights not found at {Config.MODEL_SAVE_PATH}. Using random weights."
        )

    model.to(device)
    model.eval()

    # 5. Initialize Output Canvases
    # We need to reconstruct the full predictions from tiles.
    fragment_preds = {}
    fragment_masks = {}

    # Pre-allocate canvases based on the actual fragment dimensions found by the dataset
    for fid in test_df["fragment_id"].unique():
        fid_str = str(fid)
        if fid_str in dataset.fragment_data:
            # The mask represents the valid papyrus area
            mask = dataset.fragment_data[fid_str]["mask"]
            fragment_masks[fid_str] = mask
            fragment_preds[fid_str] = np.zeros_like(mask, dtype=np.float32)
        else:
            print(f"Warning: Fragment {fid} not found in dataset cache.")

    # 6. Inference Loop
    print("Starting inference...")
    with torch.no_grad():
        for images, _, _, indices in loader:
            images = images.to(device)

            # Forward pass with Test Time Augmentation (TTA)
            # 1. Original
            outputs = model(images)
            preds = torch.sigmoid(outputs)

            if use_tta:
                # 2. Horizontal Flip
                images_h = torch.flip(images, [3])
                out_h = model(images_h)
                preds += torch.sigmoid(torch.flip(out_h, [3]))

                # 3. Vertical Flip
                images_v = torch.flip(images, [2])
                out_v = model(images_v)
                preds += torch.sigmoid(torch.flip(out_v, [2]))

                # 4. Rotate 90 (Counter-Clockwise)
                # k=1 rotates 90 deg CCW. We rotate input, predict, then rotate output back (k=-1).
                images_r = torch.rot90(images, k=1, dims=[2, 3])
                out_r = model(images_r)
                preds += torch.rot90(torch.sigmoid(out_r), k=-1, dims=[2, 3])

                # Average
                preds /= 4.0

            # Move to CPU for stitching
            preds_np = preds.cpu().numpy()

            # Stitching
            for i, idx in enumerate(indices):
                idx = idx.item()
                sample = dataset.samples[idx]

                fid = sample["fragment_id"]
                x = sample["x"]
                y = sample["y"]
                w = sample["width"]
                h = sample["height"]

                if fid not in fragment_preds:
                    continue

                full_h, full_w = fragment_preds[fid].shape

                # Calculate valid region (handling the padding added by Dataset or boundary conditions)
                # The prediction tile is (1, 512, 512).
                # The target location is (y:y+h, x:x+w).
                # We must clip to the image boundaries.
                y_end = min(y + h, full_h)
                x_end = min(x + w, full_w)

                valid_h = y_end - y
                valid_w = x_end - x

                if valid_h <= 0 or valid_w <= 0:
                    continue

                # Extract valid part of prediction
                # Prediction is (1, H, W) -> (H, W)
                pred_tile = preds_np[i, 0, :valid_h, :valid_w]

                # Place in canvas
                # Since Config.STRIDE == Config.TILE_SIZE, tiles are non-overlapping.
                fragment_preds[fid][y:y_end, x:x_end] = pred_tile

    # 7. Post-processing and Submission Generation
    submission_rows = []

    print("Generating submission file...")
    for fid in sorted(fragment_preds.keys()):
        pred_map = fragment_preds[fid]
        valid_mask = fragment_masks[fid]

        # Mask invalid areas (outside the papyrus fragment)
        pred_map = pred_map * valid_mask

        # Thresholding
        binary_map = (pred_map > threshold).astype(np.uint8)

        # Run-Length Encoding
        rle_str = rle_encoding(binary_map)

        submission_rows.append({"Id": fid, "Predicted": rle_str})

    # Save submission
    submission_df = pd.DataFrame(submission_rows)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
