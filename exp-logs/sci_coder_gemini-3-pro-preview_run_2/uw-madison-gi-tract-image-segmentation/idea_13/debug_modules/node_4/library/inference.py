import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from library.config import Config
from library.model import UNetPlusPlus25D
from library.dataset import prepare_data, GIDataset, get_transforms
from library.utils import (
    rle_encode,
    keep_largest_component,
    get_gaussian_weight_map,
)


def predict_sliding_window(model, image, device):
    """
    Performs sliding window inference on a single 2.5D image tensor.

    Args:
        model: The PyTorch model (UNetPlusPlus25D).
        image: Input tensor of shape (C, H, W).
        device: Torch device.

    Returns:
        np.ndarray: Probability map of shape (Num_Classes, H, W).
    """
    model.eval()
    image = image.to(device)
    _, H, W = image.shape

    window_h, window_w = Config.WINDOW_SIZE
    stride = Config.STRIDE

    # Calculate required padding to fit the window
    pad_h = max(0, window_h - H)
    pad_w = max(0, window_w - W)

    # Apply reflection padding
    # F.pad format: (left, right, top, bottom)
    padding = (0, pad_w, 0, pad_h)
    padded_img = F.pad(image.unsqueeze(0), padding, mode="reflect").squeeze(0)
    _, H_pad, W_pad = padded_img.shape

    # Define sliding window coordinates
    # We ensure the last window touches the bottom/right edge
    y_steps = list(range(0, H_pad - window_h + 1, stride))
    if y_steps[-1] + window_h < H_pad:
        y_steps.append(H_pad - window_h)

    x_steps = list(range(0, W_pad - window_w + 1, stride))
    if x_steps[-1] + window_w < W_pad:
        x_steps.append(W_pad - window_w)

    # Initialize accumulators for predictions and weights
    num_classes = Config.NUM_CLASSES
    pred_sum = torch.zeros((num_classes, H_pad, W_pad), device=device)
    weight_sum = torch.zeros((1, H_pad, W_pad), device=device)

    # Generate Gaussian weight map for smooth blending
    gauss_weight = torch.from_numpy(get_gaussian_weight_map((window_h, window_w))).to(
        device
    )

    # Sliding window loop
    for y in y_steps:
        for x in x_steps:
            # Extract patch
            patch = padded_img[:, y : y + window_h, x : x + window_w]

            # Predict
            with torch.no_grad():
                logits = model(patch.unsqueeze(0))

                # Handle Deep Supervision output (list of tensors)
                # We take the first element which corresponds to the final output
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]

                probs = torch.sigmoid(logits).squeeze(0)  # (C, H, W)

            # Accumulate weighted probabilities
            pred_sum[:, y : y + window_h, x : x + window_w] += probs * gauss_weight
            weight_sum[:, y : y + window_h, x : x + window_w] += gauss_weight

    # Normalize by total weight
    final_pred = pred_sum / (weight_sum + 1e-7)

    # Crop back to original image dimensions
    final_pred = final_pred[:, :H, :W]

    return final_pred.cpu().numpy()


def process_volume(volume_buffer, results_list):
    """
    Processes a buffered volume (list of slices for a single case/day),
    applies 3D post-processing (Largest Connected Component),
    encodes masks to RLE, and appends to results.

    Args:
        volume_buffer: List of tuples (id, prob_map, h, w).
        results_list: List to append result dictionaries to.
    """
    if not volume_buffer:
        return

    # Unpack buffer
    ids = [x[0] for x in volume_buffer]
    prob_maps = [x[1] for x in volume_buffer]  # List of (C, H, W) arrays

    # Stack slices to form a 3D volume: (D, C, H, W)
    # Then permute to (C, D, H, W) for class-wise processing
    volume_stack = np.stack(prob_maps, axis=1)

    # Threshold probabilities to binary masks
    mask_stack = (volume_stack > 0.5).astype(np.uint8)

    # Iterate over each class to apply 3D post-processing
    for c_idx, class_name in enumerate(Config.CLASSES):
        # Extract 3D volume for this class: (D, H, W)
        class_vol = mask_stack[c_idx]

        # Keep only the largest connected component in 3D
        processed_vol = keep_largest_component(class_vol)

        # Encode each slice back to RLE
        for d in range(processed_vol.shape[0]):
            slice_mask = processed_vol[d]
            rle = rle_encode(slice_mask)

            results_list.append({"id": ids[d], "class": class_name, "predicted": rle})


def inference():
    """
    Main inference pipeline.
    1. Loads test metadata and prepares dataset.
    2. Loads the trained model.
    3. Iterates through the dataset, grouping slices by case_day.
    4. Runs sliding window prediction and 3D post-processing.
    5. Generates and saves the submission CSV.
    """
    print("Starting Inference...")

    # 1. Load Data
    if not os.path.exists(Config.TEST_METADATA_PATH):
        print(
            f"Test metadata not found at {Config.TEST_METADATA_PATH}. Skipping inference."
        )
        return

    # Prepare data (this generates 2.5D context paths and sorts by case/day/slice)
    df_test = prepare_data(
        Config.TEST_METADATA_PATH, mode="test", load_cached_data=False
    )
    dataset = GIDataset(df_test, mode="test", transforms=get_transforms("test"))

    print(f"Test dataset size: {len(dataset)}")

    # 2. Load Model
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(
            f"Model file not found at {Config.MODEL_SAVE_PATH}. Cannot run inference."
        )
        return

    print(f"Loading model from {Config.MODEL_SAVE_PATH}...")
    model = UNetPlusPlus25D()
    model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    )
    model.to(Config.DEVICE)
    model.eval()

    # 3. Inference Loop
    results = []
    volume_buffer = []
    current_case_day = None

    # Iterate sequentially through the sorted dataset
    for i in range(len(dataset)):
        sample = dataset[i]

        # Parse ID to identify Case and Day for volume grouping
        # ID format: caseXXX_dayYY_slice_ZZZZ
        parts = sample["id"].split("_")
        case_day = f"{parts[0]}_{parts[1]}"

        # If we encounter a new volume, process the buffered one
        if current_case_day is not None and case_day != current_case_day:
            process_volume(volume_buffer, results)
            volume_buffer = []

        current_case_day = case_day

        # Run Prediction (Sliding Window)
        img_tensor = sample["image"]
        prob_map = predict_sliding_window(model, img_tensor, Config.DEVICE)

        # Buffer the result
        volume_buffer.append(
            (sample["id"], prob_map, sample["img_height"], sample["img_width"])
        )

    # Process the final volume
    if volume_buffer:
        process_volume(volume_buffer, results)

    # 4. Save Submission
    print("Saving submission...")
    submission_df = pd.DataFrame(results)

    # Ensure columns are in the correct order
    if not submission_df.empty:
        submission_df = submission_df[["id", "class", "predicted"]]
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print("Warning: No predictions generated. Submission file not created.")
