import os
import torch
import pandas as pd
import numpy as np
import cv2
import gc
from torch.utils.data import DataLoader
from library.config import Config
from library.model import BiSeNet25D
from library.dataset import UWMDataset
from library.utils import rle_encode, keep_largest_component_3d


def predict_and_submit(load_cached_data=True, debug=False):
    """
    Runs inference on the test set, applies 3D post-processing, and generates the submission file.

    Args:
        load_cached_data (bool): Whether to use cached processed data for the dataset.
        debug (bool): If True, runs on a subset of data for debugging purposes.
    """
    # 1. Setup
    Config.setup()
    device = Config.DEVICE

    # Override config debug setting if argument is provided
    Config.DEBUG = debug

    # 2. Data Preparation
    if not os.path.exists(Config.TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"Test metadata not found at {Config.TEST_METADATA_PATH}"
        )

    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Create a lookup for original image dimensions: id -> (width, height)
    # This is essential for resizing predictions back to original resolution
    meta_map = (
        test_df.drop_duplicates(subset=["id"])
        .set_index("id")[["img_width", "img_height"]]
        .to_dict("index")
    )

    # Initialize Dataset and Loader
    # UWMDataset automatically sorts by case/day/slice, allowing sequential processing
    test_ds = UWMDataset(test_df, phase="test", load_cached_data=load_cached_data)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = BiSeNet25D(num_classes=Config.NUM_CLASSES).to(device)

    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(
            f"Error: Model weights not found at {Config.MODEL_SAVE_PATH}. Please train the model first."
        )
        return

    print(f"Loading model from {Config.MODEL_SAVE_PATH}...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # 4. Inference and 3D Processing
    results = []

    # Buffer to hold all slices for a single (case, day) group
    # Structure: list of (slice_num, id, pred_mask_numpy_H_W_C)
    current_case_day = None
    case_buffer = []

    def process_buffer(buffer_list):
        """
        Helper to process a buffered volume: Stack -> 3D Post-proc -> RLE
        """
        if not buffer_list:
            return []

        # Ensure slices are sorted by slice number (Depth)
        buffer_list.sort(key=lambda x: x[0])

        slice_nums, ids, masks = zip(*buffer_list)

        # Stack to create 3D volume: (Depth, Height, Width, Classes)
        # Note: masks are (H, W, C)
        volume_3d = np.stack(masks, axis=0)

        local_results = []

        # Process each class independently
        for cls_idx, cls_name in enumerate(Config.CLASS_LABELS):
            # Extract 3D volume for specific class: (Depth, H, W)
            vol_cls = volume_3d[..., cls_idx]

            # Apply 3D connected component analysis to keep only the largest object
            # This significantly helps with Hausdorff distance by removing noise
            vol_cls_clean = keep_largest_component_3d(vol_cls)

            # Encode each slice in the cleaned volume
            for d, img_id in enumerate(ids):
                mask_slice = vol_cls_clean[d]
                rle = rle_encode(mask_slice)

                # Only add if we want to submit explicit zeros or just omit?
                # Competition usually expects all IDs or explicit empty strings.
                # rle_encode returns "" for empty masks.
                local_results.append(
                    {"id": img_id, "class": cls_name, "predicted": rle}
                )
        return local_results

    print("Running inference...")
    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            # Forward pass
            # BiSeNet returns (main, aux), we only use main for inference
            main_out, _ = model(images)

            # Sigmoid to get probabilities
            preds = torch.sigmoid(main_out).cpu().numpy()  # (B, C, 256, 256)

            # Process each image in the batch
            for i, img_id in enumerate(ids):
                # Parse ID to identify case and day groups
                # Format: caseXXX_dayYY_slice_ZZZZ
                parts = img_id.split("_")
                case_day = f"{parts[0]}_{parts[1]}"
                slice_num = int(parts[3])

                # Detect transition to a new case/day
                if current_case_day is not None and case_day != current_case_day:
                    # Process the accumulated volume for the previous case
                    results.extend(process_buffer(case_buffer))
                    case_buffer = []
                    gc.collect()  # Free memory

                current_case_day = case_day

                # Retrieve original dimensions
                orig_w = meta_map[img_id]["img_width"]
                orig_h = meta_map[img_id]["img_height"]

                # Prepare prediction for resizing
                # Transpose from (C, H, W) to (H, W, C) for OpenCV
                p_img = preds[i].transpose(1, 2, 0)  # (256, 256, 3)

                # Resize back to original resolution
                p_resized = cv2.resize(
                    p_img, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
                )

                # Binarize predictions (Threshold 0.5)
                p_bin = (p_resized > 0.5).astype(np.uint8)

                # Add to buffer
                case_buffer.append((slice_num, img_id, p_bin))

        # Process the final buffer
        if case_buffer:
            results.extend(process_buffer(case_buffer))

    # 5. Submission Generation
    sub_df = pd.DataFrame(results)

    # Ensure strict column ordering as per submission format
    sub_df = sub_df[["id", "class", "predicted"]]

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Inference complete. Submission saved to {Config.SUBMISSION_PATH}")
