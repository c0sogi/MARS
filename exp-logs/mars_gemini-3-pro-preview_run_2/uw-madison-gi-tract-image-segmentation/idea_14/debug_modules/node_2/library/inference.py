import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
from scipy.ndimage import label as scipy_label
from tqdm import tqdm

from library.config import Config
from library.model import HRNetSegmentation
from library.dataset import UWMadisonDataset
from library.utils import rle_encode


def predict_sliding_window(model, image, device):
    """
    Performs sliding window inference on a single image.

    Args:
        model: Trained PyTorch model.
        image: Input tensor of shape (C, H, W).
        device: Torch device.

    Returns:
        np.ndarray: Probability map of shape (Num_Classes, H, W).
    """
    model.eval()
    _, H, W = image.shape
    win_h, win_w = Config.INFERENCE_WINDOW_SIZE
    stride = Config.INFERENCE_STRIDE

    # Pad image to ensure it's at least the size of the window
    pad_h = max(0, win_h - H)
    pad_w = max(0, win_w - W)

    # Add batch dimension and pad
    # Padding: (left, right, top, bottom)
    img_padded = F.pad(
        image.unsqueeze(0), (0, pad_w, 0, pad_h), mode="constant", value=0
    )
    _, _, H_pad, W_pad = img_padded.shape

    # Calculate grid steps
    h_steps = []
    if H_pad <= win_h:
        h_steps = [0]
    else:
        h_steps = list(range(0, H_pad - win_h + 1, stride))
        if h_steps[-1] + win_h < H_pad:
            h_steps.append(H_pad - win_h)

    w_steps = []
    if W_pad <= win_w:
        w_steps = [0]
    else:
        w_steps = list(range(0, W_pad - win_w + 1, stride))
        if w_steps[-1] + win_w < W_pad:
            w_steps.append(W_pad - win_w)

    # Buffers for accumulation
    preds = torch.zeros((Config.NUM_CLASSES, H_pad, W_pad), device=device)
    count = torch.zeros((Config.NUM_CLASSES, H_pad, W_pad), device=device)

    with torch.no_grad():
        for y in h_steps:
            for x in w_steps:
                crop = img_padded[:, :, y : y + win_h, x : x + win_w]
                crop = crop.to(device)

                with torch.cuda.amp.autocast():
                    out = model(crop)
                    prob = torch.sigmoid(out)[0]  # (Num_Classes, win_h, win_w)

                preds[:, y : y + win_h, x : x + win_w] += prob
                count[:, y : y + win_h, x : x + win_w] += 1.0

    # Normalize and crop back to original size
    preds = preds / count
    return preds[:, :H, :W].cpu().numpy()


def post_process_volume(volume_probs, threshold=0.5):
    """
    Applies 3D Largest Connected Component filtering to a probability volume.

    Args:
        volume_probs: Numpy array of shape (D, H, W).
        threshold: Binary threshold.

    Returns:
        np.ndarray: Binary mask of shape (D, H, W).
    """
    mask = (volume_probs > threshold).astype(np.uint8)

    # Label connected components in 3D
    labeled, num_features = scipy_label(mask)

    if num_features == 0:
        return mask

    # Find largest component (ignoring background 0)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0

    if counts.max() == 0:
        return mask

    largest_label = counts.argmax()

    mask_clean = (labeled == largest_label).astype(np.uint8)
    return mask_clean


def run_inference():
    """
    Main inference routine.
    Loads data, runs prediction with sliding window, applies 3D post-processing,
    and generates the submission file.
    """
    print("Starting Inference Pipeline...")
    device = Config.DEVICE

    # 1. Load Model
    model = HRNetSegmentation(num_classes=Config.NUM_CLASSES, pretrained=False)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
        print(f"Loaded model from {Config.MODEL_SAVE_PATH}")
    else:
        print(
            f"Warning: Model file not found at {Config.MODEL_SAVE_PATH}. Using random weights."
        )

    model.to(device)
    model.eval()

    # 2. Load Test Metadata
    if not os.path.exists(Config.TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"Test metadata not found at {Config.TEST_METADATA_PATH}"
        )

    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # 3. Initialize Dataset
    # We use the dataset class to handle 2.5D loading and physical resampling
    dataset = UWMadisonDataset(df_test, phase="test", transform=None)

    results = []

    # 4. Group by Case and Day for 3D Processing
    # We process each (case, day) volume independently
    grouped = df_test.groupby(["case", "day"])

    for (case, day), group in tqdm(grouped, desc="Processing Cases"):
        # Sort slices to ensure correct 3D ordering
        group = group.sort_values("slice")
        indices = group.index.tolist()

        if not indices:
            continue

        # Initialize volume buffer based on the first slice's resampled dimensions
        first_item = dataset[indices[0]]
        _, H_res, W_res = first_item["image"].shape
        D = len(indices)

        # Buffer: (Classes, Depth, Height, Width)
        volume_preds = np.zeros((Config.NUM_CLASSES, D, H_res, W_res), dtype=np.float16)
        slice_meta = []

        # Predict slice by slice
        for z, idx in enumerate(indices):
            item = dataset[idx]
            image = item["image"].to(device)  # (3, H, W)
            orig_shape = item["orig_shape"]  # (h, w)
            img_id = item["id"]

            # Run sliding window inference
            prob_map = predict_sliding_window(model, image, device)  # (C, H, W)

            # Handle potential dimension mismatch if spacing varies within scan (rare)
            if prob_map.shape[1:] != (H_res, W_res):
                prob_map_resized = []
                for c in range(Config.NUM_CLASSES):
                    pm = cv2.resize(
                        prob_map[c], (W_res, H_res), interpolation=cv2.INTER_LINEAR
                    )
                    prob_map_resized.append(pm)
                prob_map = np.stack(prob_map_resized)

            volume_preds[:, z, :, :] = prob_map.astype(np.float16)

            slice_meta.append({"id": img_id, "orig_shape": orig_shape})

        # Post-process and Encode
        for c_idx, c_name in enumerate(Config.CLASSES):
            # Extract volume for this class
            vol = volume_preds[c_idx].astype(np.float32)

            # Apply 3D Largest Connected Component
            mask_vol = post_process_volume(vol, threshold=0.5)

            # Resize back to original dimensions and encode
            for z, meta in enumerate(slice_meta):
                mask_slice = mask_vol[z]
                orig_h, orig_w = meta["orig_shape"]

                # Resize if necessary (Nearest Neighbor for masks)
                if mask_slice.shape != (orig_h, orig_w):
                    mask_slice = cv2.resize(
                        mask_slice, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                    )

                rle = rle_encode(mask_slice)

                results.append({"id": meta["id"], "class": c_name, "predicted": rle})

    # 5. Save Submission
    submission_df = pd.DataFrame(results)
    # Ensure correct column order
    submission_df = submission_df[["id", "class", "predicted"]]
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
