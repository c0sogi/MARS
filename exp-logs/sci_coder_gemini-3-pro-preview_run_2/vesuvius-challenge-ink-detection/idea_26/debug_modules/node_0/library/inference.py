import os
import cv2
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.model import InkSegFormer
from library.data import get_test_loader
from library.utils import rle_encoding


def predict_and_submit(load_cached_data=True):
    """
    Runs the inference pipeline:
    1. Loads the best trained model.
    2. Performs Deterministic Z-Scanning (predicts on multiple Z-starts).
    3. Aggregates predictions using Max-Fusion.
    4. Generates a Run-Length Encoded submission file.

    Args:
        load_cached_data (bool): If True, attempts to use cached volume data.
    """
    # 1. Setup
    device = torch.device(Config.DEVICE)
    model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        print(
            f"Error: Model checkpoint not found at {model_path}. Cannot perform inference."
        )
        return

    print(f"Loading model from {model_path}...")
    model = InkSegFormer()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. Initialize Prediction Buffers
    # We need to know the full shape of each test fragment to allocate buffers
    test_meta_df = pd.read_csv(Config.TEST_METADATA_PATH)
    fragment_preds = {}
    fragment_shapes = {}

    print("Initializing prediction buffers...")
    for _, row in test_meta_df.iterrows():
        fid = row["fragment_id"]
        mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])

        # Read mask to get dimensions
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            print(f"Warning: Could not load mask for fragment {fid}")
            continue

        h, w = mask_img.shape
        fragment_shapes[fid] = (h, w)
        # Initialize with zeros for Max-Fusion
        fragment_preds[fid] = np.zeros((h, w), dtype=np.float32)

    # 3. Deterministic Z-Scanning Loop
    z_starts = Config.INFERENCE_Z_STARTS  # [16, 20, 24]

    with torch.no_grad():
        for z_start in z_starts:
            print(f"Running inference for Z-start: {z_start}")

            # Get loader for this specific Z-view
            loader = get_test_loader(z_start=z_start, load_cached_data=load_cached_data)
            dataset_meta = loader.dataset.metadata

            for batch_idx, (images, _) in enumerate(loader):
                images = images.to(device)

                # Predict
                logits = model(images)
                probs = torch.sigmoid(logits)  # (B, 1, 512, 512)
                probs = probs.cpu().numpy()

                # Map predictions back to full fragment canvas
                batch_size = images.size(0)
                start_idx = batch_idx * Config.BATCH_SIZE

                for i in range(batch_size):
                    # Get metadata for this specific patch
                    meta_idx = start_idx + i
                    if meta_idx >= len(dataset_meta):
                        break

                    row = dataset_meta.iloc[meta_idx]
                    fid = row["fragment_id"]
                    x, y = row["x"], row["y"]

                    if fid not in fragment_preds:
                        continue

                    full_h, full_w = fragment_shapes[fid]

                    # Determine valid region (handle edge padding)
                    # The model outputs 512x512, but if we are at the edge,
                    # the valid content might be smaller.
                    # We calculate the intersection of the patch and the full image.
                    valid_h = min(Config.TILE_SIZE, full_h - y)
                    valid_w = min(Config.TILE_SIZE, full_w - x)

                    # Extract valid region from prediction
                    pred_patch = probs[i, 0, :valid_h, :valid_w]

                    # Max-Fusion: Update the buffer with the maximum probability found so far
                    current_buffer_slice = fragment_preds[fid][
                        y : y + valid_h, x : x + valid_w
                    ]
                    fragment_preds[fid][y : y + valid_h, x : x + valid_w] = np.maximum(
                        current_buffer_slice, pred_patch
                    )

    # 4. Generate Submission
    print("Generating submission file...")
    submission_data = []

    for fid in sorted(fragment_preds.keys()):
        prob_map = fragment_preds[fid]

        # Apply mask from input to ensure we don't predict outside valid area
        # (Optional but good practice, though mask is usually handled by metric)
        # Here we just threshold.

        binary_mask = (prob_map > Config.THRESHOLD).astype(np.uint8)

        # Encode
        rle_str = rle_encoding(binary_mask)
        submission_data.append({"Id": fid, "Predicted": rle_str})

    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
