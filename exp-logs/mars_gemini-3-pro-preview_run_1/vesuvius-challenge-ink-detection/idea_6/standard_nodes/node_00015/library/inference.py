import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import InkDataset
from library.model import FFDCNet
from library.utils import seed_everything, rle_encode


def generate_submission(load_cached_data=True):
    """
    Runs inference on the test set, reconstructs full fragment masks,
    encodes them, and generates the submission.csv file.

    Args:
        load_cached_data (bool): Whether to use cached .npy files for the dataset.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # 2. Load Metadata
    if not os.path.exists(Config.TEST_METADATA):
        print(
            f"Test metadata not found at {Config.TEST_METADATA}. Cannot generate submission."
        )
        return

    test_df = pd.read_csv(Config.TEST_METADATA)

    # 3. Load Threshold
    threshold_path = os.path.join(Config.WORKING_DIR, "best_threshold.txt")
    threshold = 0.5
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            try:
                threshold = float(f.read().strip())
                print(f"Loaded best threshold: {threshold}")
            except ValueError:
                print("Error reading threshold file. Defaulting to 0.5.")
    else:
        print("Threshold file not found. Defaulting to 0.5.")

    # 4. Load Model
    model = FFDCNet().to(device)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model weights from {checkpoint_path}")
    else:
        print(
            f"Checkpoint not found at {checkpoint_path}. Using random initialization (Warning!)."
        )

    model.eval()

    # 5. Prepare Data Loader
    # shuffle=False is critical to align predictions with test_df
    test_dataset = InkDataset(Config.TEST_METADATA, load_cached_data=load_cached_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 6. Inference and Reconstruction
    # We need to reconstruct the full image for each fragment from patches.
    # Structure: fragment_predictions[fragment_id] = np.array(H, W)
    fragment_predictions = {}

    # Pre-calculate canvas sizes to avoid dynamic resizing overhead if possible
    # However, dynamic resizing is safer if metadata is sparse.
    # We will use a coordinate-based approach to determine max dimensions first.
    fragment_dims = {}
    for _, row in test_df.iterrows():
        fid = str(row["fragment_id"])
        max_x = row["x"] + row["w"]
        max_y = row["y"] + row["h"]

        if fid not in fragment_dims:
            fragment_dims[fid] = [0, 0]

        fragment_dims[fid][0] = max(fragment_dims[fid][0], max_y)
        fragment_dims[fid][1] = max(fragment_dims[fid][1], max_x)

    # Initialize canvases
    for fid, (h, w) in fragment_dims.items():
        fragment_predictions[fid] = np.zeros((h, w), dtype=np.float32)

    print("Starting inference...")

    current_idx = 0
    with torch.no_grad():
        for volumes, _ in test_loader:
            volumes = volumes.to(device, dtype=torch.float32)

            # Forward pass
            outputs = model(volumes)
            probs = torch.sigmoid(outputs)

            # Process batch
            batch_size = volumes.size(0)
            probs_np = probs.cpu().numpy()  # (B, 1, 512, 512)

            for i in range(batch_size):
                # Get metadata for this specific patch
                meta_row = test_df.iloc[current_idx]
                current_idx += 1

                fid = str(meta_row["fragment_id"])
                x = meta_row["x"]
                y = meta_row["y"]
                w = meta_row["w"]
                h = meta_row["h"]

                # Extract prediction
                # Remove channel dim: (1, 512, 512) -> (512, 512)
                pred_patch = probs_np[i, 0, :, :]

                # Crop padding (Dataset pads to 512x512, we need original w, h)
                # Padding is added to bottom and right
                pred_valid = pred_patch[:h, :w]

                # Place in canvas
                fragment_predictions[fid][y : y + h, x : x + w] = pred_valid

    # 7. Generate Submission
    print("Encoding predictions...")
    submission_rows = []

    # Sort fragment IDs to ensure consistent output order (though not strictly required)
    sorted_fids = sorted(fragment_predictions.keys())

    for fid in sorted_fids:
        pred_map = fragment_predictions[fid]

        # Apply threshold
        binary_mask = (pred_map >= threshold).astype(np.uint8)

        # RLE Encode
        rle_str = rle_encode(binary_mask)

        submission_rows.append({"Id": fid, "Predicted": rle_str})

    # Create DataFrame and save
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
