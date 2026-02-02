import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import cv2
from scipy import ndimage

from library.config import Config
from library.utils import set_seed, rle_encode
from library.dataset import process_metadata, UWDataset, get_transforms
from library.model import LinkNet


def predict_volume(model, loader, device):
    """
    Generates raw probability maps for the test set.
    Resizes predictions back to original image dimensions.

    Args:
        model (torch.nn.Module): The trained model.
        loader (DataLoader): DataLoader for the test set.
        device (str): Device to run inference on.

    Returns:
        list: A list of dictionaries, each containing 'id' and 'prediction' (C, H, W).
    """
    model.eval()
    preds_cache = []

    with torch.no_grad():
        for images, ids, sizes in loader:
            images = images.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)

            # Move to CPU for processing
            outputs = outputs.cpu().numpy()

            for i in range(len(ids)):
                slice_id = ids[i]
                # sizes is a tensor batch of [H, W]
                orig_h, orig_w = sizes[i]
                pred_mask = outputs[i]  # Shape: (C, H, W)

                # Resize back to original size
                # Transpose to (H, W, C) for cv2.resize
                pred_mask = np.transpose(pred_mask, (1, 2, 0))

                # cv2.resize expects (width, height)
                pred_mask = cv2.resize(pred_mask, (int(orig_w), int(orig_h)))

                # Transpose back to (C, H, W)
                pred_mask = np.transpose(pred_mask, (2, 0, 1))

                preds_cache.append({"id": slice_id, "prediction": pred_mask})

    return preds_cache


def post_process_3d(preds_cache, threshold=0.5, use_3d_cc=True):
    """
    Reconstructs 3D volumes from 2D predictions, applies Connected Component Analysis,
    and generates RLE encodings.

    Args:
        preds_cache (list): List of dicts with predictions.
        threshold (float): Threshold for binarization.
        use_3d_cc (bool): Whether to apply 3D connected component filtering.

    Returns:
        pd.DataFrame: Dataframe formatted for submission.
    """

    # Helper to parse ID for grouping
    def get_case_day(row_id):
        # id format: case123_day20_slice_0001
        parts = row_id.split("_")
        case = parts[0]
        day = parts[1]
        # slice number is the 4th part (index 3)
        slice_num = int(parts[3])
        return f"{case}_{day}", slice_num

    # Group by case_day
    case_groups = {}
    for item in preds_cache:
        cd, sn = get_case_day(item["id"])
        if cd not in case_groups:
            case_groups[cd] = []
        case_groups[cd].append((sn, item))

    final_submission = []

    # Process each case volume
    for case_id, items in case_groups.items():
        # Sort by slice number to form a proper volume
        items.sort(key=lambda x: x[0])

        # Get dimensions from the first slice
        sample_pred = items[0][1]["prediction"]
        C, H, W = sample_pred.shape
        D = len(items)

        # Create volume (C, D, H, W)
        volume = np.zeros((C, D, H, W), dtype=np.float32)
        slice_ids = []

        for z, (sn, item) in enumerate(items):
            volume[:, z, :, :] = item["prediction"]
            slice_ids.append(item["id"])

        # Threshold to binary
        volume_bin = (volume > threshold).astype(np.uint8)

        # 3D Connected Components per class
        if use_3d_cc:
            for c in range(C):
                class_vol = volume_bin[c]
                if class_vol.sum() > 0:
                    # Label connected components
                    labeled, num_features = ndimage.label(class_vol)
                    if num_features > 1:
                        # Find largest component
                        sizes = ndimage.sum(
                            class_vol, labeled, range(1, num_features + 1)
                        )
                        largest_label = np.argmax(sizes) + 1
                        # Keep only largest
                        class_vol = (labeled == largest_label).astype(np.uint8)
                        volume_bin[c] = class_vol

        # Encode RLE for each slice in the volume
        for z in range(D):
            current_id = slice_ids[z]
            for c, class_name in enumerate(Config.CLASS_LABELS):
                mask_slice = volume_bin[c, z, :, :]
                rle = rle_encode(mask_slice)
                final_submission.append(
                    {"id": current_id, "class": class_name, "predicted": rle}
                )

    # Convert to DataFrame
    sub_df = pd.DataFrame(final_submission)
    return sub_df


def run_inference(load_cached_data=True):
    """
    Main entry point for inference.
    Loads data, model, runs prediction, post-processing, and saves submission.
    """
    set_seed(Config.SEED)

    # 1. Load Data
    print("Processing metadata for inference...")
    # process_metadata handles caching internally based on file existence,
    # but we pass the flag as requested by the prompt guidelines.
    test_df = process_metadata(
        Config.TEST_METADATA_PATH, mode="test", load_cached_data=load_cached_data
    )

    test_dataset = UWDataset(test_df, mode="test", transforms=get_transforms("test"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE * 2,  # Inference can handle larger batches
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Load Model
    if not os.path.exists(Config.MODEL_PATH):
        print(f"Model file not found at {Config.MODEL_PATH}. Cannot run inference.")
        return

    print(f"Loading model from {Config.MODEL_PATH}...")
    model = LinkNet().to(Config.DEVICE)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))

    # 3. Predict
    print("Running prediction on test set...")
    preds_cache = predict_volume(model, test_loader, Config.DEVICE)

    # 4. Post-processing
    print("Running 3D post-processing...")
    sub_df = post_process_3d(
        preds_cache,
        threshold=Config.PRED_THRESHOLD,
        use_3d_cc=Config.USE_3D_CONNECTED_COMPONENTS,
    )

    # 5. Save Submission
    # Ensure correct column order
    if not sub_df.empty:
        sub_df = sub_df[["id", "class", "predicted"]]
    else:
        # Handle edge case of empty predictions (should not happen with valid test set)
        sub_df = pd.DataFrame(columns=["id", "class", "predicted"])

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
