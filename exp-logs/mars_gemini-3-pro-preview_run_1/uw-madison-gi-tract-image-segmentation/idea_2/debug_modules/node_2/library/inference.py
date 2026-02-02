import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, rle_encode, keep_largest_component_3d
from library.model import UNet25D
from library.dataset import UWGIDataset, get_transforms


def run_inference():
    """
    Executes the inference pipeline:
    1. Loads test metadata and prepares the dataset.
    2. Loads the trained model checkpoint.
    3. Generates predictions slice-by-slice.
    4. Aggregates slices into 3D volumes per case/day.
    5. Applies 3D post-processing (keeping largest connected component).
    6. Resizes masks to original dimensions and encodes (RLE).
    7. Saves the submission file.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Initializing inference on device: {device}")

    # 2. Load Metadata
    if not os.path.exists(Config.TEST_CSV):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_CSV}")

    test_df = pd.read_csv(Config.TEST_CSV, keep_default_na=False)

    # Create a lookup for original dimensions: id -> (width, height)
    # This is necessary because the model outputs fixed size (320x320)
    # but submission requires original resolution.
    dims_lookup = test_df.set_index("id")[["width", "height"]].to_dict("index")

    # 3. Dataset and Loader
    # We use the same dataset class as training to ensure consistent preprocessing (2.5D stacking)
    test_dataset = UWGIDataset(test_df, transforms=get_transforms("test"), mode="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE * 2,  # Inference can handle larger batches
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Load Model
    model = UNet25D(
        backbone_name=Config.BACKBONE,
        classes=Config.NUM_CLASSES,
        pretrained=False,  # No need to download weights, we load checkpoint
    )
    model = model.to(device)

    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Please train the model first."
        )

    print(f"Loading model weights from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 5. Prediction Loop
    # We need to aggregate predictions by case_day to perform 3D post-processing
    # Structure: results_dict[case_day] = list of {'slice_num': int, 'id': str, 'mask': np.array}
    results_dict = {}

    print("Generating predictions...")

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device, dtype=torch.float32)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Thresholding
            preds = (probs > 0.5).float().cpu().numpy()

            # Organize by case/day
            for i, slice_id in enumerate(ids):
                # slice_id format: caseXXX_dayYY_slice_ZZZZ
                parts = slice_id.split("_")
                case_day = f"{parts[0]}_{parts[1]}"
                slice_num = int(parts[3])

                if case_day not in results_dict:
                    results_dict[case_day] = []

                results_dict[case_day].append(
                    {
                        "slice_num": slice_num,
                        "id": slice_id,
                        "mask": preds[i],  # Shape: (C, H, W)
                    }
                )

    # 6. Post-processing and Encoding
    print("Applying 3D post-processing and encoding...")
    submission_rows = []

    for case_day, slices in results_dict.items():
        # Sort slices by slice number to ensure correct Z-ordering
        slices.sort(key=lambda x: x["slice_num"])

        # Stack into 4D volume: (Num_Slices, C, H, W)
        vol_stack = np.stack([s["mask"] for s in slices], axis=0)

        # Transpose to (C, Num_Slices, H, W) -> (C, D, H, W)
        vol_stack = vol_stack.transpose(1, 0, 2, 3)

        # Process each class channel independently
        processed_vol = np.zeros_like(vol_stack)

        for c in range(Config.NUM_CLASSES):
            # Extract 3D volume for this class
            class_vol = vol_stack[c]  # (D, H, W)

            # Apply 3D Connected Component Analysis (Keep Largest)
            # This removes small noise artifacts which hurt Hausdorff distance
            cleaned_vol = keep_largest_component_3d(class_vol)

            processed_vol[c] = cleaned_vol

        # Iterate back through the slices to resize and encode
        # Transpose back to (Num_Slices, C, H, W) for iteration
        processed_vol = processed_vol.transpose(1, 0, 2, 3)

        for i, s_info in enumerate(slices):
            slice_id = s_info["id"]
            orig_w = dims_lookup[slice_id]["width"]
            orig_h = dims_lookup[slice_id]["height"]

            for c_idx, class_name in enumerate(Config.CLASSES):
                # Get the processed mask for this slice and class
                mask_slice = processed_vol[i, c_idx, :, :]

                # Resize back to original resolution
                # Use nearest neighbor interpolation to maintain binary values (0 or 1)
                mask_resized = cv2.resize(
                    mask_slice, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                )

                # Run-Length Encoding
                rle = rle_encode(mask_resized)

                submission_rows.append(
                    {"id": slice_id, "class": class_name, "predicted": rle}
                )

    # 7. Save Submission
    sub_df = pd.DataFrame(submission_rows)

    # Ensure columns are in correct order
    sub_df = sub_df[["id", "class", "predicted"]]

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_FILE, index=False)

    print(f"Inference complete. Submission saved to {Config.SUBMISSION_FILE}")
