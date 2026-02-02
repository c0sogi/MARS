import torch
import numpy as np
import pandas as pd
import cv2
import os
from pathlib import Path
from library.config import Config
from library.model import ResidualFCN
from library.data import get_dataloaders
from library.utils import rle_encode


def predict_batch(model, images):
    """
    Performs a simple inference step on a batch of images.
    """
    with torch.no_grad():
        preds = model(images)
    return preds


def inference_tta(model, images):
    """
    Performs Test-Time Augmentation (TTA) inference.

    Applies 8 geometric transformations (D4 group: 4 rotations * 2 flips),
    runs inference, inverts the transformations, and averages the results.

    Args:
        model: The trained PyTorch model.
        images: Batch of images (B, C, H, W).

    Returns:
        Tensor of averaged predictions (B, 1, H, W).
    """
    batch_size, c, h, w = images.shape
    preds_accum = torch.zeros((batch_size, 1, h, w), device=images.device)

    # Define transformations: (k_rot, do_flip)
    # k_rot: number of 90-degree rotations (0, 1, 2, 3)
    # do_flip: boolean, whether to flip horizontally before rotation
    transforms = [
        (0, False),
        (1, False),
        (2, False),
        (3, False),
        (0, True),
        (1, True),
        (2, True),
        (3, True),
    ]

    # We use only a subset if TTA_STEPS is restricted, but usually we want all 8 for D4
    # Config.TTA_STEPS is set to 8 in the provided config.

    cnt = 0
    for k, do_flip in transforms:
        if cnt >= Config.TTA_STEPS:
            break

        # 1. Augment
        x = images.clone()
        if do_flip:
            x = torch.flip(x, dims=[3])  # Flip width
        if k > 0:
            x = torch.rot90(x, k=k, dims=[2, 3])

        # 2. Inference
        y = predict_batch(model, x)

        # 3. Inverse Transform
        if k > 0:
            y = torch.rot90(y, k=-k, dims=[2, 3])
        if do_flip:
            y = torch.flip(y, dims=[3])

        preds_accum += y
        cnt += 1

    return preds_accum / cnt


def generate_submission(
    checkpoint_path=Config.MODEL_PATH,
    output_path=Config.SUBMISSION_PATH,
    batch_size=Config.BATCH_SIZE,
):
    """
    Generates the submission.csv file for the competition.

    Iterates over test fragments, performs tiled inference with TTA,
    stitches the results, applies thresholding and RLE encoding.
    """

    # 1. Setup
    device = Config.DEVICE

    # Load Model
    model = ResidualFCN().to(device)

    if os.path.exists(checkpoint_path):
        # Load weights
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Model checkpoint not found at {checkpoint_path}. Using random weights."
        )

    model.eval()

    # Load Data
    # We need the test metadata to know dimensions and IDs
    if not Config.TEST_METADATA_PATH.exists():
        print("Test metadata not found. Cannot generate submission.")
        return

    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Get loaders (returns dict: fragment_id -> loader)
    dataloaders_dict = get_dataloaders(Config)
    if "test_dict" not in dataloaders_dict:
        print("No test dataloaders found.")
        return

    test_loaders = dataloaders_dict["test_dict"]

    submission_data = []

    # 2. Inference Loop per Fragment
    for _, row in df_test.iterrows():
        fragment_id = str(row["fragment_id"])

        if fragment_id not in test_loaders:
            continue

        loader = test_loaders[fragment_id]

        # Dimensions
        H, W = row["height"], row["width"]

        # Initialize canvas for reconstruction
        # We use float32 for accumulation
        prob_map = torch.zeros((H, W), dtype=torch.float32, device="cpu")
        count_map = torch.zeros((H, W), dtype=torch.float32, device="cpu")

        # Iterate over tiles
        # We disable tqdm for silent execution as requested
        for images, _, coords in loader:
            images = images.to(device)

            # Predict with TTA
            with torch.no_grad():
                preds = inference_tta(model, images)  # (B, 1, h, w)

            preds = preds.squeeze(1).cpu()  # (B, h, w)

            # Place on canvas
            for i in range(images.size(0)):
                y, x = coords[i]
                h_crop, w_crop = preds[i].shape

                prob_map[y : y + h_crop, x : x + w_crop] += preds[i]
                count_map[y : y + h_crop, x : x + w_crop] += 1.0

        # 3. Normalize and Post-process
        # Avoid division by zero
        count_map[count_map == 0] = 1.0
        prob_map /= count_map

        # Convert to numpy
        prob_map = prob_map.numpy()

        # Apply valid mask
        # The metadata contains 'mask_path'. We should mask out non-fragment pixels.
        mask_path = row.get("mask_path")
        if pd.notna(mask_path):
            full_mask_path = Config.INPUT_DIR / mask_path
            if full_mask_path.exists():
                valid_mask = cv2.imread(str(full_mask_path), cv2.IMREAD_GRAYSCALE)
                if valid_mask is not None:
                    # Resize if necessary (though shapes should match)
                    if valid_mask.shape != prob_map.shape:
                        valid_mask = cv2.resize(
                            valid_mask, (W, H), interpolation=cv2.INTER_NEAREST
                        )

                    prob_map = prob_map * (valid_mask > 0)

        # Threshold
        binary_pred = (prob_map > Config.THRESHOLD).astype(np.uint8)

        # RLE Encode
        rle_str = rle_encode(binary_pred)

        submission_data.append({"Id": fragment_id, "Predicted": rle_str})

        # Clear memory
        del prob_map, count_map, binary_pred
        torch.cuda.empty_cache()

    # 4. Save Submission
    df_sub = pd.DataFrame(submission_data)
    df_sub.to_csv(output_path, index=False)
