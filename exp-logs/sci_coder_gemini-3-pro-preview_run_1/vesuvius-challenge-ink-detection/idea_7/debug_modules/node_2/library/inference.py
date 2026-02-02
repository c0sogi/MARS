import os
import torch
import numpy as np
import pandas as pd
from library import config, dataset, model, utils


def generate_submission(load_cached_data=True):
    """
    Generates the submission file by:
    1. Loading the trained model.
    2. Optimizing the decision threshold on the validation set.
    3. Predicting on the test set and stitching patches.
    4. Encoding results and saving to CSV.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data from disk.
    """
    # Ensure reproducibility
    utils.set_seed(config.SEED)

    print(f"Initializing inference on device: {config.DEVICE}")

    # --- 1. Load Model ---
    net = model.HDNet().to(config.DEVICE)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Train the model first."
        )

    net.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    net.eval()

    # --- 2. Optimize Threshold (Validation Phase) ---
    print("Loading dataloaders...")
    # We only need val and test loaders here
    _, val_loader, test_loader = dataset.get_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        load_cached_data=load_cached_data,
    )

    print("Running inference on validation set for threshold optimization...")
    val_probs = []
    val_targets = []

    with torch.no_grad():
        for volumes, labels in val_loader:
            volumes = volumes.to(config.DEVICE)

            # Forward pass
            outputs = net(volumes)
            probs = torch.sigmoid(outputs)

            # Collect results
            val_probs.append(probs.cpu().numpy())
            val_targets.append(labels.cpu().numpy())

    # Concatenate all batches
    y_true_val = np.concatenate(val_targets)
    y_probs_val = np.concatenate(val_probs)

    # Find best threshold
    # Note: utils.optimize_threshold prints the best score in full precision
    best_threshold = utils.optimize_threshold(y_true_val, y_probs_val, beta=0.5)

    # --- 3. Test Inference & Stitching ---
    print(f"Running inference on test set with threshold {best_threshold}...")

    # Load test metadata to determine canvas dimensions and patch coordinates
    df_test = pd.read_csv(config.TEST_METADATA_PATH)

    # Pre-calculate fragment dimensions
    fragment_dims = {}
    for fid, group in df_test.groupby("fragment_id"):
        max_x = (group["x"] + group["w"]).max()
        max_y = (group["y"] + group["h"]).max()
        fragment_dims[fid] = (max_y, max_x)  # Height, Width

    # Initialize blank canvases for reconstruction
    fragment_masks = {
        fid: np.zeros(dims, dtype=np.uint8) for fid, dims in fragment_dims.items()
    }

    # Create a lookup for patch metadata: sample_id -> (fragment_id, x, y, w, h)
    meta_lookup = {}
    for _, row in df_test.iterrows():
        meta_lookup[row["sample_id"]] = (
            row["fragment_id"],
            row["x"],
            row["y"],
            row["w"],
            row["h"],
        )

    with torch.no_grad():
        for volumes, sample_ids in test_loader:
            volumes = volumes.to(config.DEVICE)

            outputs = net(volumes)
            probs = torch.sigmoid(outputs)

            # Apply threshold
            preds = (probs >= best_threshold).float().cpu().numpy()

            # Stitch patches onto canvases
            for i, sample_id in enumerate(sample_ids):
                # Retrieve metadata
                fid, x, y, w, h = meta_lookup[sample_id]

                # Extract prediction (remove channel dim: 1, H, W -> H, W)
                pred_patch = preds[i, 0]

                # Crop padding
                # The dataset pads to (PATCH_SIZE, PATCH_SIZE) on the bottom and right.
                # Valid data is in the top-left corner [0:h, 0:w].
                valid_patch = pred_patch[:h, :w]

                # Place on canvas
                # Note: Overlapping patches (if any) will overwrite.
                # Given stride=patch_size in metadata generation, this is a direct tile placement.
                fragment_masks[fid][y : y + h, x : x + w] = valid_patch.astype(np.uint8)

    # --- 4. RLE Encoding & Submission ---
    print("Encoding predictions and generating submission file...")
    predictions = {}
    for fid, mask in fragment_masks.items():
        rle = utils.rle_encode(mask)
        predictions[fid] = rle

    utils.write_submission(predictions)
    print("Inference complete.")
