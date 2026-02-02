import os
import torch
import numpy as np
import pandas as pd
from library.config import (
    CHECKPOINT_DIR,
    SUBMISSION_PATH,
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    PATCH_SIZE,
)
from library.model import ParallelDilatedCNN
from library.dataset import get_dataloaders
from library.utils import calibrate_threshold, rle_encode


def predict_and_submit(
    checkpoint_path=os.path.join(CHECKPOINT_DIR, "best_model.pth"),
    submission_path=SUBMISSION_PATH,
    device=DEVICE,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    load_cached_data=True,
):
    """
    Performs inference using the trained model.

    Steps:
    1. Loads the model state from checkpoint.
    2. Runs inference on the Validation set to calibrate the binary threshold (optimizing F0.5).
    3. Runs inference on the Test set, stitching patches into full fragment masks.
    4. Applies the calibrated threshold and RLE encoding.
    5. Saves the result to submission.csv.
    """

    # 1. Setup and Load Model
    print(f"Loading model from {checkpoint_path}...")
    model = ParallelDilatedCNN().to(device)

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Checkpoint file {checkpoint_path} not found. Using random initialization (for debugging only)."
        )

    model.eval()

    # 2. Get DataLoaders
    # We need validation data for threshold calibration and test data for submission
    _, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        load_cached_data=load_cached_data,
    )

    # 3. Threshold Calibration
    # If validation data exists, find the threshold that maximizes F0.5.
    # Otherwise, fallback to 0.5.
    best_threshold = 0.5

    if len(val_loader) > 0:
        print("Running validation inference for threshold calibration...")
        val_probs_list = []
        val_targets_list = []

        with torch.no_grad():
            for data, target, _ in val_loader:
                data = data.to(device)

                # Forward pass
                output = model(data)
                probs = torch.sigmoid(output)

                # Store results (move to CPU to save GPU memory)
                val_probs_list.append(probs.cpu().numpy())
                val_targets_list.append(target.numpy())

        # Concatenate all batches
        val_probs = np.concatenate(val_probs_list)
        val_targets = np.concatenate(val_targets_list)

        # Calibrate
        best_threshold, best_score = calibrate_threshold(
            val_targets, val_probs, beta=0.5
        )
        print(
            f"Calibration Result: Threshold={best_threshold}, Validation F0.5={best_score}"
        )
    else:
        print("No validation data found. Using default threshold 0.5.")

    # 4. Test Inference & Stitching
    print("Running test inference...")

    # Access metadata to determine full fragment dimensions
    test_metadata = test_loader.dataset.metadata

    if test_metadata.empty:
        print("Warning: Test metadata is empty. Creating empty submission.")
        pd.DataFrame(columns=["Id", "Predicted"]).to_csv(submission_path, index=False)
        return

    # Create a lookup for fast metadata access by sample_id
    meta_lookup = test_metadata.set_index("sample_id").to_dict("index")

    # Initialize canvases for each fragment
    fragment_ids = test_metadata["fragment_id"].unique()
    fragment_masks = {}

    for fid in fragment_ids:
        f_df = test_metadata[test_metadata["fragment_id"] == fid]
        # Calculate dimensions required to hold all patches
        max_w = (f_df["x"] + f_df["w"]).max()
        max_h = (f_df["y"] + f_df["h"]).max()
        fragment_masks[fid] = np.zeros((max_h, max_w), dtype=np.float32)

    # Inference Loop
    with torch.no_grad():
        for data, _, sample_ids in test_loader:
            data = data.to(device)
            output = model(data)
            probs = torch.sigmoid(output).cpu().numpy()

            # Place each patch onto the corresponding fragment canvas
            for i, sample_id in enumerate(sample_ids):
                # Retrieve patch metadata
                info = meta_lookup[sample_id]
                fid = info["fragment_id"]
                x, y, w, h = info["x"], info["y"], info["w"], info["h"]

                # Extract valid area from prediction
                # The model outputs PATCH_SIZE x PATCH_SIZE (e.g., 512x512)
                # If the patch was at the edge, it was padded. We only take the top-left w x h.
                pred_patch = probs[i, 0, :h, :w]

                # Assign to canvas
                fragment_masks[fid][y : y + h, x : x + w] = pred_patch

    # 5. Encode and Save
    print(f"Encoding predictions with threshold {best_threshold}...")
    submission_data = []

    for fid in sorted(fragment_ids):
        prob_map = fragment_masks[fid]

        # Apply threshold
        binary_mask = (prob_map >= best_threshold).astype(np.uint8)

        # Run-Length Encode
        rle_str = rle_encode(binary_mask)

        submission_data.append({"Id": fid, "Predicted": rle_str})

    # Write to CSV
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
