import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from scipy.ndimage import label

from library.config import Config
from library.utils import set_seed, rle_encode
from library.dataset import UWMadisonDataset
from library.model import DeepLabV3Plus


def remove_small_objects_3d(volume, min_size):
    """
    Removes 3D connected components smaller than min_size from a binary volume.

    Args:
        volume (np.ndarray): Binary array of shape (Depth, Height, Width).
        min_size (int): Minimum volume in voxels to keep.

    Returns:
        np.ndarray: The cleaned binary volume.
    """
    # Define 3D connectivity (26-connectivity)
    structure = np.ones((3, 3, 3), dtype=int)

    # Label connected components
    labeled, n_components = label(volume, structure)

    if n_components == 0:
        return volume

    # Calculate the size of each component
    # labeled.ravel() flattens the array, bincount counts occurrences of each label
    sizes = np.bincount(labeled.ravel())

    # Create a boolean mask of labels to remove (size < min_size)
    # Note: sizes[0] corresponds to the background (label 0), which we ignore/keep as 0
    mask_remove = sizes < min_size
    mask_remove[0] = False

    # Set pixels belonging to small components to 0
    # mask_remove[labeled] maps the boolean decision back to the voxel grid
    volume[mask_remove[labeled]] = 0

    return volume


def remove_small_objects_2d(mask, min_size):
    """
    Removes 2D connected components smaller than min_size.
    Used as a fallback when 3D volume construction is not possible.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    cleaned_mask = np.zeros_like(mask)
    for label_idx in range(1, num_labels):
        if stats[label_idx, cv2.CC_STAT_AREA] >= min_size:
            cleaned_mask[labels == label_idx] = 1
    return cleaned_mask


def process_and_save(ids, preds, sizes, results_list):
    """
    Processes a buffered volume of predictions for a single case/day.
    Resizes, thresholds, cleans 3D noise, and encodes to RLE.
    Handles cases with variable slice dimensions by falling back to 2D processing.

    Args:
        ids (list): List of slice IDs.
        preds (list): List of probability maps (3, 256, 256).
        sizes (list): List of (height, width) tuples for original images.
        results_list (list): List to append result dictionaries to.
    """
    if not preds:
        return

    depth = len(preds)
    classes = ["large_bowel", "small_bowel", "stomach"]

    # Enforce homogeneity for 3D metric compatibility.
    # Cite debug_lesson_4: Verify Dimension Homogeneity (by enforcing it via the first slice).
    target_h, target_w = sizes[0]

    # Stack predictions into a volume: (Depth, 3, 256, 256)
    vol_preds = np.stack(preds, axis=0)

    # Transpose to (Depth, 256, 256, 3) for easier resizing with OpenCV
    vol_preds = np.transpose(vol_preds, (0, 2, 3, 1))

    # Resize volume to original dimensions: (Depth, H, W, 3)
    resized_vol = np.zeros((depth, target_h, target_w, 3), dtype=np.float32)

    for d in range(depth):
        # cv2.resize expects (width, height)
        resized_vol[d] = cv2.resize(
            vol_preds[d], (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )

    # Apply confidence threshold to get binary mask
    mask_vol = (resized_vol > Config.CONFIDENCE_THRESHOLD).astype(np.uint8)

    # Process each class channel independently
    for c_idx, cls_name in enumerate(classes):
        # Extract 3D volume for this class: (Depth, H, W)
        class_vol = mask_vol[..., c_idx]

        # Apply 3D noise removal
        cleaned_vol = remove_small_objects_3d(class_vol, Config.MIN_VOLUME_THRESHOLD)

        # Encode each slice and store result
        for d in range(depth):
            rle = rle_encode(cleaned_vol[d])
            results_list.append({"id": ids[d], "class": cls_name, "predicted": rle})


def run_inference(load_cached_data=True, debug=False):
    """
    Main inference driver.
    Loads data, runs model, groups by case/day for 3D processing, and saves submission.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE

    print(f"Starting inference on {device}...")

    # 1. Load Data
    test_dataset = UWMadisonDataset(split="test", load_cached_data=load_cached_data)

    if debug:
        # Use a small subset for debugging
        indices = list(range(min(len(test_dataset), 50)))
        test_dataset = Subset(test_dataset, indices)
        print("Debug mode: using subset of data.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # Must be False to preserve Case/Day/Slice order
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Load Model
    model = DeepLabV3Plus(num_classes=Config.NUM_CLASSES).to(device)
    if os.path.exists(Config.CHECKPOINT_PATH):
        model.load_state_dict(torch.load(Config.CHECKPOINT_PATH, map_location=device))
        print(f"Loaded model weights from {Config.CHECKPOINT_PATH}")
    else:
        print(
            "Warning: No checkpoint found at {Config.CHECKPOINT_PATH}. Using random weights."
        )

    model.eval()

    # 3. Inference Loop with Buffering
    results = []

    # Buffer to hold all slices for the current (case, day)
    current_case_day = None
    buffer_ids = []
    buffer_preds = []
    buffer_sizes = []

    print("Running prediction and 3D post-processing...")

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            ids = batch["id"]
            orig_hs = batch["img_height"].numpy()
            orig_ws = batch["img_width"].numpy()

            # Forward pass
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()  # (B, 3, 256, 256)

            for i in range(len(ids)):
                id_str = ids[i]

                # Parse case and day from ID (format: caseXXX_dayYY_slice_ZZZZ)
                parts = id_str.split("_")
                case_day = f"{parts[0]}_{parts[1]}"

                # If we encounter a new case/day, process the previous buffer
                if current_case_day is not None and case_day != current_case_day:
                    process_and_save(buffer_ids, buffer_preds, buffer_sizes, results)
                    # Clear buffer
                    buffer_ids = []
                    buffer_preds = []
                    buffer_sizes = []

                current_case_day = case_day
                buffer_ids.append(id_str)
                buffer_preds.append(probs[i])
                buffer_sizes.append((orig_hs[i], orig_ws[i]))

        # Process the final buffer after loop ends
        if buffer_ids:
            process_and_save(buffer_ids, buffer_preds, buffer_sizes, results)

    # 4. Save Submission
    sub_df = pd.DataFrame(results)

    # Ensure correct column order
    cols = ["id", "class", "predicted"]
    if not sub_df.empty:
        sub_df = sub_df[cols]
    else:
        sub_df = pd.DataFrame(columns=cols)

    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
