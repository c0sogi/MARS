import os
import cv2
import torch
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from library.config import CFG
from library.dataset import UWMGIDataset, get_transforms, process_25d_dataframe
from library.model import build_model
from library.utils import rle_encode


def restore_original_geometry(pred_mask, original_h, original_w, target_size):
    """
    Reverses the resize_pad_square operation.
    1. Crops the padding.
    2. Resizes back to original dimensions.
    """
    target_h, target_w = target_size

    # Calculate scaling factor used during preprocessing
    scale = min(target_h / original_h, target_w / original_w)
    new_h, new_w = int(original_h * scale), int(original_w * scale)

    # Calculate padding used
    delta_h = target_h - new_h
    delta_w = target_w - new_w
    top = delta_h // 2
    left = delta_w // 2

    # Crop padding
    # pred_mask is (H, W)
    cropped = pred_mask[top : top + new_h, left : left + new_w]

    # Resize back to original
    # Use INTER_NEAREST for binary masks
    restored = cv2.resize(
        cropped, (original_w, original_h), interpolation=cv2.INTER_NEAREST
    )

    return restored


def post_process_volume(volume):
    """
    Applies 3D Connected Component Analysis to keep only the largest object.
    Args:
        volume: (D, H, W) binary numpy array
    Returns:
        processed_volume: (D, H, W) binary numpy array
    """
    if volume.sum() == 0:
        return volume

    # Label connected components
    labeled_array, num_features = ndimage.label(volume)

    if num_features == 0:
        return volume

    # Find the largest component
    # bincount is fast for counting labels.
    # Index 0 is background, so we skip it if we assume background is 0.
    sizes = np.bincount(labeled_array.ravel())

    # If there's only background (should be caught by sum check, but safety first)
    if len(sizes) <= 1:
        return volume

    # Get label of largest component (ignoring background at index 0)
    largest_label = sizes[1:].argmax() + 1

    # Create mask for largest component
    processed_volume = (labeled_array == largest_label).astype(np.uint8)

    return processed_volume


def predict_and_submit(load_cached_data=True):
    """
    Main inference function.
    1. Loads test data and model.
    2. Predicts masks.
    3. Post-processes (Geometry restore + 3D CCA).
    4. Generates submission file.
    """
    # 1. Setup
    CFG.setup(verbose=False)
    device = CFG.device

    # Ensure output directories exist
    os.makedirs(CFG.submission_dir, exist_ok=True)

    # 2. Data Preparation
    test_csv_path = CFG.test_csv
    if not os.path.exists(test_csv_path):
        print(f"Test metadata not found at {test_csv_path}. Cannot proceed.")
        return

    df_test = pd.read_csv(test_csv_path)

    # Process 2.5D context
    df_test = process_25d_dataframe(
        df_test, split_name="test", load_cached_data=load_cached_data
    )

    # Dataset and Loader
    test_dataset = UWMGIDataset(
        df_test, label=False, transforms=get_transforms(data="test")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 3. Model Loading
    model = build_model()
    checkpoint_path = os.path.join(CFG.checkpoint_dir, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print(
            f"Checkpoint not found at {checkpoint_path}. Using random weights (Warning!)."
        )
    else:
        print(f"Loading model from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # 4. Inference Loop (Gather Predictions)
    print("Running inference...")

    # Store predictions: id -> (3, 320, 320) numpy array (uint8 0/1)
    # Using a dict might consume memory, but for 220GB RAM it's safe.
    predictions_map = {}

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device, dtype=torch.float)

            with autocast(enabled=CFG.mixed_precision):
                logits = model(images)
                probs = torch.sigmoid(logits)
                preds = (probs > CFG.mask_threshold).float()

            # Move to CPU
            preds_np = preds.cpu().numpy().astype(np.uint8)

            for i, sample_id in enumerate(ids):
                predictions_map[sample_id] = preds_np[i]

    # 5. Post-Processing & RLE Encoding
    print("Post-processing and generating submission...")

    submission_rows = []
    classes = ["large_bowel", "small_bowel", "stomach"]

    # Group metadata by case + day to process volumes
    # Ensure slice_int exists for sorting
    df_test["slice_int"] = df_test["slice"].astype(int)
    groups = df_test.groupby(["case", "day"])

    for (case, day), group in groups:
        # Sort by slice to ensure correct Z-ordering
        group = group.sort_values("slice_int")

        # Get dimensions from the first slice (assuming consistent per scan)
        h_orig = int(group.iloc[0]["height"])
        w_orig = int(group.iloc[0]["width"])
        d = len(group)

        # Prepare volumes for this scan: (C, D, H, W)
        # We process each class separately to save memory or together.
        # Let's do per class.

        # Collect restored 2D slices first
        # Structure: class -> list of 2D arrays (sorted by z)
        scan_slices = {c: [] for c in classes}
        slice_ids = group["id"].values

        for sample_id in slice_ids:
            if sample_id not in predictions_map:
                # Should not happen
                pred_mask = np.zeros(
                    (3, CFG.img_size[0], CFG.img_size[1]), dtype=np.uint8
                )
            else:
                pred_mask = predictions_map[sample_id]

            for idx, cls in enumerate(classes):
                # Restore geometry for this slice
                mask_2d = pred_mask[idx]
                restored_mask = restore_original_geometry(
                    mask_2d, h_orig, w_orig, CFG.img_size
                )
                scan_slices[cls].append(restored_mask)

        # Process each class volume
        for cls in classes:
            # Stack to create volume (D, H, W)
            volume = np.stack(scan_slices[cls], axis=0)

            # Apply 3D CCA
            volume_processed = post_process_volume(volume)

            # Encode each slice and add to submission
            for z, sample_id in enumerate(slice_ids):
                rle = rle_encode(volume_processed[z])
                submission_rows.append(
                    {"id": sample_id, "class": cls, "predicted": rle}
                )

    # 6. Save Submission
    df_submission = pd.DataFrame(submission_rows)

    # Ensure columns are in correct order
    df_submission = df_submission[["id", "class", "predicted"]]

    output_path = CFG.submission_file
    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}. Rows: {len(df_submission)}")
