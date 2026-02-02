import os
import cv2
import torch
import numpy as np
import pandas as pd
from scipy import ndimage
from collections import defaultdict
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, rle_encode
from library.ghost_model import GhostUNet
from library.dataset import process_metadata, UWGI25DDataset, get_transforms


def post_process_3d(volume):
    """
    Applies 3D connected component analysis to a volume of shape (Depth, H, W, C).
    Retains only the largest connected component for each class to reduce Hausdorff distance error.
    """
    processed_vol = np.zeros_like(volume)

    # Iterate over classes
    for c in range(volume.shape[3]):
        class_vol = volume[..., c]
        if class_vol.sum() == 0:
            continue

        # Label connected components
        # Structure defines connectivity (default is 3x3x3 cross)
        labeled, num_features = ndimage.label(class_vol)

        if num_features > 0:
            # Calculate size of each component
            # sizes[0] corresponds to background (label 0), so we ignore it
            sizes = ndimage.sum(class_vol, labeled, range(num_features + 1))

            # Find label of largest component (excluding background)
            if len(sizes) > 1:
                largest_label = sizes[1:].argmax() + 1
                # Keep only pixels belonging to the largest component
                processed_vol[..., c] = (labeled == largest_label).astype(np.uint8)

    return processed_vol


def run_inference():
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference using device: {device}")

    # ==========================================
    # 1. Load Data
    # ==========================================
    # Process test metadata to get file paths and 2.5D context
    test_df = process_metadata(
        Config.TEST_METADATA_PATH, "test_processed", load_cached_data=True
    )

    # Create Dataset and DataLoader
    # We use a larger batch size for inference as gradients are not required
    test_ds = UWGI25DDataset(test_df, transforms=get_transforms("test"), mode="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # Create a lookup for original image dimensions: id -> (height, width)
    shape_lookup = test_df.set_index("id")[["img_height", "img_width"]].to_dict("index")

    # ==========================================
    # 2. Load Model
    # ==========================================
    model = GhostUNet(in_channels=Config.IN_CHANNELS, num_classes=Config.NUM_CLASSES)

    if os.path.exists(Config.MODEL_PATH):
        print(f"Loading model from {Config.MODEL_PATH}")
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    else:
        print(
            f"Warning: Model file not found at {Config.MODEL_PATH}. Using random weights."
        )

    model.to(device)
    model.eval()

    # ==========================================
    # 3. Prediction Loop
    # ==========================================
    # Structure: preds_map[case_day][slice_num] = {'mask': np.array, 'id': str}
    preds_map = defaultdict(dict)

    print("Running prediction loop...")
    with torch.no_grad():
        for images, _, ids in test_loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            # Convert to numpy: (Batch, Channels, Height, Width)
            probs_np = probs.cpu().numpy()

            for i, img_id in enumerate(ids):
                # Parse ID to group by case and day
                # ID Format: caseXXX_dayYY_slice_ZZZZ
                parts = img_id.split("_")
                case_day = f"{parts[0]}_{parts[1]}"
                # Extract slice number (assuming format ..._slice_0001)
                try:
                    slice_num = int(parts[3])
                except (IndexError, ValueError):
                    # Fallback if ID format is unexpected
                    slice_num = i

                # Threshold predictions to binary mask
                # Transpose to (H, W, C) for easier 3D stacking later
                mask = (probs_np[i] > 0.5).astype(np.uint8).transpose(1, 2, 0)

                preds_map[case_day][slice_num] = {"mask": mask, "id": img_id}

    # ==========================================
    # 4. Post-processing & RLE Generation
    # ==========================================
    submission_rows = []

    print("Post-processing 3D volumes and encoding RLE...")

    # Process each case_day group
    for case_day, slices_dict in preds_map.items():
        # Sort slices by number to ensure correct 3D volume construction
        sorted_slice_nums = sorted(slices_dict.keys())

        # Stack slices to form 3D volume: (Depth, H, W, C)
        # Note: These are all 256x256 at this point
        volume = np.stack([slices_dict[s]["mask"] for s in sorted_slice_nums])

        # Apply 3D Connected Component Analysis
        refined_volume = post_process_3d(volume)

        # Iterate back through the refined slices to generate submission entries
        for idx, slice_num in enumerate(sorted_slice_nums):
            slice_data = slices_dict[slice_num]
            img_id = slice_data["id"]

            # Get the refined mask slice: (H, W, C)
            refined_mask = refined_volume[idx]

            # Get original dimensions
            orig_h = shape_lookup[img_id]["img_height"]
            orig_w = shape_lookup[img_id]["img_width"]

            # Process each class
            for cls_idx, cls_name in enumerate(Config.CLASSES):
                # Extract binary mask for the class
                mask_cls = refined_mask[..., cls_idx]

                # Resize back to original image dimensions
                # cv2.resize expects (width, height)
                if (mask_cls.shape[0] != orig_h) or (mask_cls.shape[1] != orig_w):
                    mask_cls = cv2.resize(
                        mask_cls, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                    )

                # Encode to RLE
                rle = rle_encode(mask_cls)

                # Append to results
                submission_rows.append(
                    {"id": img_id, "class": cls_name, "predicted": rle}
                )

    # ==========================================
    # 5. Save Submission
    # ==========================================
    # Create DataFrame from predictions
    pred_df = pd.DataFrame(submission_rows)

    # Load sample submission to ensure all IDs are present and order is correct
    sample_sub_path = os.path.join(Config.INPUT_DIR, "sample_submission.csv")
    if os.path.exists(sample_sub_path):
        sample_sub = pd.read_csv(sample_sub_path)
        # Drop the 'predicted' column from sample if it exists to avoid conflict
        if "predicted" in sample_sub.columns:
            sample_sub = sample_sub.drop(columns=["predicted"])

        # Merge predictions into sample submission structure
        # Left join ensures we keep all rows from sample_sub
        final_sub = sample_sub.merge(pred_df, on=["id", "class"], how="left")

        # Fill missing predictions with empty string
        final_sub["predicted"] = final_sub["predicted"].fillna("")
    else:
        # Fallback if sample submission is missing (unlikely)
        final_sub = pred_df

    # Save to CSV
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    final_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
