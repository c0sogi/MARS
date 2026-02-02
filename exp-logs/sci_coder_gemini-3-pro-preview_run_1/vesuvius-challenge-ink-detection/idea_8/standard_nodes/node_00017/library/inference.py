import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import InkDataset
from library.model import FRUNet
from library.utils import rle_encode


def predict_and_encode(
    checkpoint_path=None,
    threshold_path=None,
    output_file=None,
    load_cached_data=True,
    batch_size=None,
    num_workers=None,
):
    """
    Runs inference on the test set using the trained FR-UNet model, reconstructs
    fragment masks, applies RLE encoding, and saves the submission file.

    Args:
        checkpoint_path (str, optional): Path to the model weights. Defaults to Config.CHECKPOINT_DIR/best_model.pth.
        threshold_path (str, optional): Path to the saved threshold. Defaults to Config.CHECKPOINT_DIR/best_threshold.txt.
        output_file (str, optional): Path to save the submission CSV. Defaults to Config.SUBMISSION_FILE.
        load_cached_data (bool): Whether to use cached preprocessed data. Defaults to True.
        batch_size (int, optional): Batch size for inference. Defaults to Config.BATCH_SIZE.
        num_workers (int, optional): Number of worker threads. Defaults to Config.NUM_WORKERS.
    """
    # --- 1. Setup Defaults ---
    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if threshold_path is None:
        threshold_path = os.path.join(Config.CHECKPOINT_DIR, "best_threshold.txt")

    if output_file is None:
        output_file = Config.SUBMISSION_FILE

    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    if num_workers is None:
        num_workers = Config.NUM_WORKERS

    device = torch.device(Config.DEVICE)
    print(f"Starting inference on device: {device}")

    # --- 2. Load Model ---
    model = FRUNet().to(device)

    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(
            f"Warning: Checkpoint not found at {checkpoint_path}. Using random initialization (likely to fail)."
        )

    model.eval()

    # --- 3. Load Threshold ---
    threshold = 0.5
    if os.path.exists(threshold_path):
        try:
            with open(threshold_path, "r") as f:
                threshold = float(f.read().strip())
            print(f"Loaded optimized threshold: {threshold}")
        except Exception as e:
            print(f"Error loading threshold file: {e}. Using default 0.5.")
    else:
        print("Threshold file not found. Using default 0.5.")

    # --- 4. Data Loading ---
    # Ensure test metadata exists
    if not os.path.exists(Config.TEST_CSV):
        print("Test metadata not found. Skipping inference.")
        return

    test_dataset = InkDataset(mode="test", load_cached_data=load_cached_data)

    # Handle empty test set case
    if len(test_dataset) == 0:
        print("Test dataset is empty. Generating empty submission.")
        pd.DataFrame(columns=["Id", "Predicted"]).to_csv(output_file, index=False)
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Create a lookup for original patch dimensions
    sample_dims = test_dataset.df.set_index("sample_id")[["w", "h"]].to_dict("index")

    # --- 5. Inference Loop ---
    # Dictionary to store patch predictions: fragment_id -> list of (x, y, w, h, pred_mask)
    fragment_preds = {}

    print("Running prediction loop...")
    with torch.no_grad():
        for volumes, _, _, sample_ids in test_loader:
            volumes = volumes.to(device)

            # Forward pass
            outputs = model(volumes)

            # Apply threshold immediately to save memory (keep as float 0.0/1.0 or bool)
            # We use float for easier handling downstream
            preds = (outputs >= threshold).float().cpu().numpy()

            for i, sample_id in enumerate(sample_ids):
                # Parse sample_id: {fragment_id}_{y}_{x}
                parts = sample_id.split("_")
                fid = parts[0]
                y = int(parts[1])
                x = int(parts[2])

                # Retrieve original dimensions
                dims = sample_dims[sample_id]
                orig_w = dims["w"]
                orig_h = dims["h"]

                # Crop the padded prediction back to original valid size
                pred_valid = preds[i, 0, :orig_h, :orig_w]

                if fid not in fragment_preds:
                    fragment_preds[fid] = []

                fragment_preds[fid].append((x, y, orig_w, orig_h, pred_valid))

    # --- 6. Reconstruction and Encoding ---
    print("Reconstructing fragments and encoding...")
    submission_rows = []

    # Sort keys to ensure deterministic order
    for fid in sorted(fragment_preds.keys()):
        patches = fragment_preds[fid]

        # Determine full fragment size
        # Try to read the mask file from input to get exact dimensions
        mask_path = os.path.join(Config.INPUT_DIR, "test", fid, "mask.png")
        if os.path.exists(mask_path):
            img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            H, W = img.shape
        else:
            # Fallback: infer from patch coordinates
            max_x = 0
            max_y = 0
            for p in patches:
                max_x = max(max_x, p[0] + p[2])
                max_y = max(max_y, p[1] + p[3])
            W, H = max_x, max_y
            print(
                f"Warning: Mask file not found for fragment {fid}. Inferred size: {W}x{H}"
            )

        # Create canvas (uint8 for RLE encoding)
        full_mask = np.zeros((H, W), dtype=np.uint8)

        # Paste patches
        for x, y, w, h, pred in patches:
            # Ensure we don't go out of bounds (though crop logic should prevent this)
            h_end = min(y + h, H)
            w_end = min(x + w, W)
            h_eff = h_end - y
            w_eff = w_end - x

            full_mask[y:h_end, x:w_end] = pred[:h_eff, :w_eff].astype(np.uint8)

        # Run-Length Encode
        rle = rle_encode(full_mask)
        submission_rows.append({"Id": fid, "Predicted": rle})

    # --- 7. Save Submission ---
    if submission_rows:
        df = pd.DataFrame(submission_rows)
        df.to_csv(output_file, index=False)
        print(f"Submission saved to {output_file}")
    else:
        # Create empty submission if no predictions were generated
        pd.DataFrame(columns=["Id", "Predicted"]).to_csv(output_file, index=False)
        print(
            f"Warning: No predictions generated. Empty submission saved to {output_file}"
        )
