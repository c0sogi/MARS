import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from library.config import Config
from library.model import get_model
from library.data import load_fragment_mips
from library.utils import rle_encoding


def predict_fragment(model, fragment_id, volume_path, mask_path, device, tta=True):
    """
    Performs inference on a single fragment using sliding window and TTA.

    Args:
        model (torch.nn.Module): The loaded model.
        fragment_id (str): ID of the fragment.
        volume_path (str): Relative path to the volume directory.
        mask_path (str): Relative path to the mask file.
        device (torch.device): Computation device.
        tta (bool): Whether to apply Test Time Augmentation.

    Returns:
        np.ndarray: Binary prediction mask (0 or 1).
    """
    # 1. Load Data
    # mips shape: (C, H, W)
    mips = load_fragment_mips(fragment_id, volume_path, load_cached_data=True)

    # Load mask for valid pixels
    full_mask_path = os.path.join(Config.INPUT_DIR, mask_path)
    valid_mask = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)

    if valid_mask is None:
        # Fallback if mask is missing, though unlikely in valid dataset
        _, h, w = mips.shape
        valid_mask = np.ones((h, w), dtype=np.uint8) * 255

    # Normalize to [0, 1] as per training
    # Convert to float32
    image = mips.astype(np.float32) / Config.PIXEL_MAX

    c, h, w = image.shape

    # 2. Padding
    # Pad image to be divisible by TILE_SIZE
    pad_h = (Config.TILE_SIZE - (h % Config.TILE_SIZE)) % Config.TILE_SIZE
    pad_w = (Config.TILE_SIZE - (w % Config.TILE_SIZE)) % Config.TILE_SIZE

    if pad_h > 0 or pad_w > 0:
        image = np.pad(
            image, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0
        )

    padded_h, padded_w = image.shape[1], image.shape[2]

    # Initialize probability map
    prob_map = np.zeros((padded_h, padded_w), dtype=np.float32)

    # 3. Sliding Window Inference
    # Using STRIDE same as TILE_SIZE (non-overlapping) for efficiency as per config,
    # but TTA provides the smoothing.

    model.eval()

    # Pre-calculate steps
    y_steps = range(0, padded_h, Config.STRIDE)
    x_steps = range(0, padded_w, Config.STRIDE)

    with torch.no_grad():
        for y in y_steps:
            for x in x_steps:
                # Extract tile: (C, H, W)
                tile = image[:, y : y + Config.TILE_SIZE, x : x + Config.TILE_SIZE]

                # To Tensor: (1, C, H, W)
                tile_t = torch.from_numpy(tile).unsqueeze(0).to(device)

                if tta:
                    # Create batch of augmentations
                    # 1. Original
                    # 2. Horizontal Flip
                    # 3. Vertical Flip
                    # 4. Rotate 90
                    t_h = torch.flip(tile_t, [3])
                    t_v = torch.flip(tile_t, [2])
                    t_r = torch.rot90(tile_t, 1, [2, 3])

                    batch = torch.cat([tile_t, t_h, t_v, t_r], dim=0)

                    # Inference
                    outputs = model(batch)
                    logits = outputs.logits

                    # Upsample (SegFormer outputs 1/4 res)
                    logits = F.interpolate(
                        logits,
                        size=(Config.TILE_SIZE, Config.TILE_SIZE),
                        mode="bilinear",
                        align_corners=False,
                    )

                    probs = torch.sigmoid(logits)

                    # Split batch
                    p_orig, p_h, p_v, p_r = torch.chunk(probs, 4, dim=0)

                    # Inverse Transforms
                    p_h = torch.flip(p_h, [3])
                    p_v = torch.flip(p_v, [2])
                    p_r = torch.rot90(p_r, -1, [2, 3])

                    # Average
                    avg_prob = (p_orig + p_h + p_v + p_r) / 4.0
                    tile_prob = avg_prob[0, 0].cpu().numpy()

                else:
                    # No TTA
                    outputs = model(tile_t)
                    logits = outputs.logits
                    logits = F.interpolate(
                        logits,
                        size=(Config.TILE_SIZE, Config.TILE_SIZE),
                        mode="bilinear",
                        align_corners=False,
                    )
                    probs = torch.sigmoid(logits)
                    tile_prob = probs[0, 0].cpu().numpy()

                # Place in map
                prob_map[y : y + Config.TILE_SIZE, x : x + Config.TILE_SIZE] = tile_prob

    # 4. Crop and Mask
    # Crop back to original size
    prob_map = prob_map[:h, :w]

    # Apply valid mask (set invalid areas to 0)
    prob_map[valid_mask == 0] = 0

    # Threshold
    binary_mask = (prob_map > 0.5).astype(np.uint8)

    return binary_mask


def generate_submission(
    test_metadata_path=Config.TEST_METADATA_PATH,
    checkpoint_path=Config.CHECKPOINT_PATH,
    output_path=Config.SUBMISSION_PATH,
):
    """
    Main inference routine. Generates submission.csv for the test set.

    Args:
        test_metadata_path (str): Path to test.csv.
        checkpoint_path (str): Path to the trained model weights.
        output_path (str): Path to save the submission file.
    """
    print("Initializing Inference Pipeline...")
    device = Config.DEVICE

    # Load Metadata
    if not os.path.exists(test_metadata_path):
        raise FileNotFoundError(f"Test metadata not found at {test_metadata_path}")

    df_test = pd.read_csv(test_metadata_path)
    print(f"Found {len(df_test)} test fragments.")

    # Load Model
    model = get_model()
    model.to(device)

    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"WARNING: Checkpoint not found at {checkpoint_path}.")
        print(
            "Using random weights. This usually indicates training failed to beat baseline."
        )
        # In a real competition, we might abort or submit zeros, but here we proceed.

    submission_data = []

    for idx, row in df_test.iterrows():
        frag_id = row["fragment_id"]
        vol_path = row["volume_path"]
        mask_path = row["mask_path"]

        print(f"Processing fragment {frag_id}...")

        try:
            # Predict
            pred_mask = predict_fragment(
                model, frag_id, vol_path, mask_path, device, tta=True
            )

            # Encode
            rle_str = rle_encoding(pred_mask)
            submission_data.append({"Id": frag_id, "Predicted": rle_str})

        except Exception as e:
            print(f"Error processing fragment {frag_id}: {e}")
            # Append empty prediction on failure to keep submission valid format
            submission_data.append({"Id": frag_id, "Predicted": ""})

    # Save Submission
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
