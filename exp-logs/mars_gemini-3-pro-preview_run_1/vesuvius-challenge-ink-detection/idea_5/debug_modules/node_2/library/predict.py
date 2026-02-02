import os
import random
import numpy as np
import pandas as pd
import torch
from library import config, utils, model, dataset


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_patch_dimensions(df):
    """
    Creates a lookup dictionary for valid patch dimensions.
    Args:
        df (pd.DataFrame): Test metadata.
    Returns:
        dict: Mapping of sample_id to (width, height).
    """
    dims = {}
    for _, row in df.iterrows():
        dims[row["sample_id"]] = (row["w"], row["h"])
    return dims


def get_fragment_canvases(df):
    """
    Initializes zero-filled canvases for each fragment in the test set.
    Args:
        df (pd.DataFrame): Test metadata.
    Returns:
        dict: Mapping of fragment_id to numpy array (H, W).
    """
    fragment_sizes = {}
    for _, row in df.iterrows():
        fid = row["fragment_id"]
        # Calculate the maximum extent of the fragment
        max_x = row["x"] + row["w"]
        max_y = row["y"] + row["h"]

        if fid not in fragment_sizes:
            fragment_sizes[fid] = [0, 0]

        fragment_sizes[fid][0] = max(fragment_sizes[fid][0], max_x)
        fragment_sizes[fid][1] = max(fragment_sizes[fid][1], max_y)

    canvases = {}
    for fid, (w, h) in fragment_sizes.items():
        # Initialize with float32 for probability accumulation
        canvases[fid] = np.zeros((h, w), dtype=np.float32)

    return canvases


def run_inference(
    checkpoint_path=os.path.join(config.CHECKPOINT_DIR, "best_model.pth"),
    threshold_path=os.path.join(config.WORKING_DIR, "best_threshold.txt"),
    output_file=config.SUBMISSION_FILE,
    limit=None,
):
    """
    Executes the inference pipeline.

    Args:
        checkpoint_path (str): Path to the trained model weights.
        threshold_path (str): Path to the file containing the optimized threshold.
        output_file (str): Path to save the submission CSV.
        limit (int, optional): Limit the number of samples for debugging.
    """
    # 1. Setup
    config.setup_directories()
    set_seed(config.SEED)
    device = config.DEVICE

    # 2. Load Threshold
    threshold = 0.5
    if os.path.exists(threshold_path):
        try:
            with open(threshold_path, "r") as f:
                content = f.read().strip()
                if content:
                    threshold = float(content)
        except Exception:
            pass  # Keep default if read fails

    # 3. Prepare Data
    # Load metadata to determine canvas sizes and patch validity
    test_df = pd.read_csv(config.TEST_METADATA)
    if limit is not None:
        test_df = test_df.iloc[:limit]

    patch_dims = get_patch_dimensions(test_df)
    canvases = get_fragment_canvases(test_df)

    # Get DataLoader
    # We ignore train/val loaders
    _, _, test_loader = dataset.get_dataloaders(
        batch_size=config.BATCH_SIZE, limit=limit
    )

    # 4. Load Model
    net = model.SFRPNet().to(device)
    if os.path.exists(checkpoint_path):
        utils.load_checkpoint(net, checkpoint_path)
    else:
        # If no checkpoint exists (e.g. cold start), we cannot predict meaningfully.
        # However, to prevent crash, we proceed with initialized weights (random prediction).
        pass

    net.eval()

    # 5. Prediction Loop
    with torch.no_grad():
        for batch in test_loader:
            vol = batch["volume"].to(device)
            sample_ids = batch["sample_id"]
            fragment_ids = batch["fragment_id"]
            xs = batch["x"]
            ys = batch["y"]

            # Forward pass
            logits = net(vol)
            probs = torch.sigmoid(logits)
            probs_np = probs.cpu().numpy()  # Shape: (B, 1, H, W)

            # Stitching
            for i, sample_id in enumerate(sample_ids):
                fid = fragment_ids[i]
                x = xs[i].item()
                y = ys[i].item()

                # Extract prediction map (H, W)
                pred_map = probs_np[i, 0, :, :]

                # Crop to valid dimensions (removing padding added by dataset)
                if sample_id in patch_dims:
                    w, h = patch_dims[sample_id]
                    valid_pred = pred_map[:h, :w]

                    # Place on canvas
                    # Metadata generation ensures non-overlapping patches (stride=size),
                    # so direct assignment is valid.
                    canvases[fid][y : y + h, x : x + w] = valid_pred

    # 6. Generate Submission
    submission_data = []

    for fid in sorted(canvases.keys()):
        canvas = canvases[fid]

        # Apply threshold to create binary mask
        binary_mask = (canvas > threshold).astype(np.uint8)

        # Run-Length Encode
        rle_str = utils.rle_encode(binary_mask)
        submission_data.append({"Id": fid, "Predicted": rle_str})

    # Save to CSV
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(output_file, index=False)
