import os
import cv2
import torch
import numpy as np
import pandas as pd
import scipy.ndimage
from library.config import Config
from library.utils import rle_encode, set_seed
from library.model import AttentionUNet25D
from library.dataset import get_test_loader


def keep_largest_component(volume):
    """
    Keeps only the largest connected component in a 3D binary volume.

    Args:
        volume (np.ndarray): 3D binary array (Depth, Height, Width).

    Returns:
        np.ndarray: Processed 3D binary array.
    """
    # Label connected components
    # structure=None defaults to a connectivity of 1 (squared connectivity for 3D)
    labeled_array, num_features = scipy.ndimage.label(volume)

    if num_features == 0:
        return volume

    # Calculate size of each component
    # bincount returns count of each label value (0..num_features)
    # 0 is background, so we skip it
    sizes = np.bincount(labeled_array.ravel())

    # If only background exists
    if len(sizes) < 2:
        return volume

    # Find label of largest component (skipping index 0 which is background)
    largest_label = sizes[1:].argmax() + 1

    # Create mask for largest component
    new_volume = (labeled_array == largest_label).astype(np.uint8)

    return new_volume


def predict_and_submit(load_cached_data=True):
    """
    Runs inference on the test set, applies 3D post-processing,
    and generates the submission file.

    Args:
        load_cached_data (bool): Whether to use cached metadata/loaders.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Model
    print(f"Loading model from {Config.MODEL_SAVE_PATH}...")
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_SAVE_PATH}. Train the model first."
        )

    model = AttentionUNet25D()
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 2. Prepare Data
    print("Loading test data...")
    test_loader = get_test_loader(load_cached_data=load_cached_data)

    # Load metadata for original dimensions
    # We need this to resize predictions back to the original image size
    # Use the dataframe from the loader which was dynamically scanned
    df_test_meta = test_loader.dataset.df
    # Create a lookup for dimensions: id -> (width, height)
    dims_lookup = df_test_meta.set_index("id")[["img_width", "img_height"]].to_dict(
        "index"
    )

    # Container for predictions: grouped by case_day
    # Structure: { 'case_day': { slice_num: numpy_array_of_shape_(3, 256, 256) } }
    predictions_map = {}

    print("Running inference...")
    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Binarize (Thresholding)
            preds = (probs > Config.THRESHOLD).float().cpu().numpy().astype(np.uint8)

            # Store in map
            for i, img_id in enumerate(ids):
                # Parse ID to extract grouping info
                # Format: case123_day20_slice_0001
                parts = img_id.split("_")
                case_day = f"{parts[0]}_{parts[1]}"
                slice_num = int(parts[3])

                if case_day not in predictions_map:
                    predictions_map[case_day] = {}

                predictions_map[case_day][slice_num] = preds[i]

    # 3. Post-Processing (3D) and RLE Generation
    print("Applying 3D post-processing and generating RLE...")
    submission_data = []
    classes = ["large_bowel", "small_bowel", "stomach"]

    # Iterate over each case
    for case_day, slices_dict in predictions_map.items():
        # Sort slices by number to ensure correct 3D stacking order
        sorted_slice_nums = sorted(slices_dict.keys())

        # Stack slices: (Depth, Channels, Height, Width)
        # Height and Width are 256 here (model output size)
        volume_stack = np.stack([slices_dict[s] for s in sorted_slice_nums], axis=0)

        # Process each class independently
        for c_idx, class_name in enumerate(classes):
            # Extract 3D volume for this class: (Depth, Height, Width)
            class_volume = volume_stack[:, c_idx, :, :]

            # Apply 3D Connected Component Analysis
            if Config.KEEP_LARGEST_COMPONENT:
                class_volume = keep_largest_component(class_volume)

            # Iterate back through slices to resize and encode
            for d_idx, slice_num in enumerate(sorted_slice_nums):
                # Reconstruct ID
                slice_id = f"{case_day}_slice_{slice_num:04d}"

                # Get original dimensions from metadata
                if slice_id in dims_lookup:
                    orig_w = dims_lookup[slice_id]["img_width"]
                    orig_h = dims_lookup[slice_id]["img_height"]
                else:
                    # Fallback (should not happen if metadata is complete)
                    orig_w, orig_h = Config.IMG_SIZE, Config.IMG_SIZE

                # Get the processed mask for this slice
                mask_256 = class_volume[d_idx]

                # Resize to original dimensions if needed
                if (Config.IMG_SIZE != orig_h) or (Config.IMG_SIZE != orig_w):
                    # Resize using Nearest Neighbor to maintain binary values
                    # cv2.resize expects (width, height)
                    mask_orig = cv2.resize(
                        mask_256, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                    )
                else:
                    mask_orig = mask_256

                # Encode
                rle = rle_encode(mask_orig)

                submission_data.append(
                    {"id": slice_id, "class": class_name, "predicted": rle}
                )

    # 4. Save Submission
    df_submission = pd.DataFrame(submission_data)

    # Ensure columns are in correct order
    df_submission = df_submission[["id", "class", "predicted"]]

    # Sort by ID and Class for consistency
    df_submission.sort_values(by=["id", "class"], inplace=True)

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(
        f"Submission saved to {Config.SUBMISSION_PATH} with {len(df_submission)} rows."
    )
