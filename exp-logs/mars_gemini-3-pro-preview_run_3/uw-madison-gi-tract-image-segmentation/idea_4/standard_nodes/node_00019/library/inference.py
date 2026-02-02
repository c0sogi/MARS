import os
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from library.config import (
    DEVICE,
    TEST_CSV,
    CHECKPOINT_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_CLASSES,
    THR_LARGE_BOWEL,
    THR_SMALL_BOWEL,
    THR_STOMACH,
    CLASS_LABELS,
    SEED,
)
from library.utils import set_seed, rle_encode, keep_largest_component_3d
from library.dataset import prepare_data, UWDataset, get_transforms
from library.model import UnetPlusPlus


def predict_and_submit(
    test_csv_path=TEST_CSV,
    checkpoint_path=os.path.join(CHECKPOINT_DIR, "best_model.pth"),
    output_dir="./submission",
    output_filename="submission.csv",
):
    """
    Generates predictions for the test set, applies 3D post-processing,
    and saves the submission file in RLE format.
    """
    # 1. Setup
    set_seed(SEED)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    print(f"Starting inference using model: {checkpoint_path}")

    # 2. Data Preparation
    # prepare_data handles the 2.5D logic (prev/next paths) and caching
    # We force load_cached_data=True to use existing cache if available,
    # but the function will recompute if the cache is missing.
    df_test = prepare_data(test_csv_path, mode="test", load_cached_data=True)

    test_dataset = UWDataset(
        df_test, mode="test", transform=get_transforms(mode="test")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # 3. Model Loading
    model = UnetPlusPlus().to(DEVICE)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    # Handle state dict keys if they were saved with 'model_state_dict' wrapper or directly
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    # 4. Inference Loop
    # We store predictions in RAM. For 6800 slices * 3 classes * 320 * 384 * 4 bytes ~= 10GB.
    # This fits comfortably in the provided 220GB RAM.
    all_preds = []

    print("Running inference on test batches...")
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(DEVICE, dtype=torch.float32)

            with autocast():
                # Model returns logits
                logits = model(images)
                probs = torch.sigmoid(logits)

            # Move to CPU numpy
            all_preds.append(probs.cpu().numpy())

    # Concatenate all batches: (N, C, H_model, W_model)
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
    else:
        # Handle empty test set case
        all_preds = np.zeros((0, NUM_CLASSES, 320, 384))

    # 5. Volume Reconstruction & Post-processing
    submission_rows = []

    # Thresholds map
    thresholds = [THR_LARGE_BOWEL, THR_SMALL_BOWEL, THR_STOMACH]

    # Group by Case + Day to form volumes
    # df_test is aligned with all_preds index-wise because prepare_data sorts and resets index
    groups = df_test.groupby(["case", "day"])

    print(f"Processing {len(groups)} volumes for 3D CCA and RLE encoding...")

    for (case, day), group_df in groups:
        # Get indices for this volume
        indices = group_df.index.values

        # Extract volume predictions: (D, C, H_model, W_model)
        vol_preds = all_preds[indices]

        # Ensure slices are sorted spatially (top to bottom) based on slice number
        slice_nums = group_df["slice"].values
        sort_idx = np.argsort(slice_nums)

        # Apply sorting
        vol_preds = vol_preds[sort_idx]
        sorted_indices = indices[sort_idx]

        # Get original dimensions for this scan (assuming constant per scan)
        # We check the first slice in the sorted group
        first_idx = sorted_indices[0]
        orig_h = df_test.loc[first_idx, "height"]
        orig_w = df_test.loc[first_idx, "width"]

        # Process each class
        for cls_idx, cls_name in enumerate(CLASS_LABELS):
            # 1. Extract class volume: (D, H_model, W_model)
            cls_vol_probs = vol_preds[:, cls_idx, :, :]

            # 2. Threshold
            cls_vol_mask = (cls_vol_probs > thresholds[cls_idx]).astype(np.uint8)

            # 3. 3D CCA (Keep largest component)
            cls_vol_processed = keep_largest_component_3d(cls_vol_mask)

            # 4. Process each slice for submission
            for i, global_idx in enumerate(sorted_indices):
                # Get the processed mask for this slice: (H_model, W_model)
                slice_mask = cls_vol_processed[i]

                # Resize to original dimensions if necessary
                # Use Nearest Neighbor to keep it binary
                if (slice_mask.shape[0] != orig_h) or (slice_mask.shape[1] != orig_w):
                    slice_mask = cv2.resize(
                        slice_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                    )

                # RLE Encode
                rle = rle_encode(slice_mask)

                # Construct ID
                slice_id = df_test.loc[global_idx, "id"]

                submission_rows.append(
                    {"id": slice_id, "class": cls_name, "predicted": rle}
                )

    # 6. Save Submission
    df_submission = pd.DataFrame(submission_rows)

    # Ensure columns order
    df_submission = df_submission[["id", "class", "predicted"]]

    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Total rows: {len(df_submission)}")
