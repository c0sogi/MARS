import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.model import PSDN
from library.dataset import InkDataset
from library.utils import rle_encoding


def apply_tta(images):
    """
    Applies Test-Time Augmentation (TTA) to a batch of images.
    Returns a list of tuples: (augmented_images, inverse_transform_func).
    Strategies: Identity, Horizontal Flip, Vertical Flip, Rot90.
    """
    transforms = []

    # 1. Identity
    transforms.append((images, lambda x: x))

    # 2. Horizontal Flip (dim 3 is width)
    transforms.append((torch.flip(images, dims=[3]), lambda x: torch.flip(x, dims=[3])))

    # 3. Vertical Flip (dim 2 is height)
    transforms.append((torch.flip(images, dims=[2]), lambda x: torch.flip(x, dims=[2])))

    # 4. Rotate 90 degrees (k=1)
    # Input is (B, C, H, W), rotation is on dims (2, 3)
    transforms.append(
        (
            torch.rot90(images, k=1, dims=[2, 3]),
            lambda x: torch.rot90(x, k=3, dims=[2, 3]),  # Inverse is k=3 (270 deg)
        )
    )

    return transforms


def predict_fragment(model, loader, device):
    """
    Generates probability maps for all fragments in the loader using TTA.
    Returns a dictionary mapping fragment_idx to {'prob': map, 'count': map}.
    """
    model.eval()
    dataset = loader.dataset

    # Initialize storage for results
    # Key: fragment_index (int) -> Value: {'prob': np.array, 'count': np.array}
    results = {}

    # Initialize maps based on dataset metadata
    # Note: The dataset fragments are padded, so we initialize maps to the padded size.
    for i, frag in enumerate(dataset.fragments):
        h, w = frag["mask"].shape
        results[i] = {
            "prob": np.zeros((h, w), dtype=np.float32),
            "count": np.zeros((h, w), dtype=np.float32),
        }

    print(f"Starting inference on {len(dataset)} patches...")

    with torch.no_grad():
        for volumes, meta in loader:
            # volumes: (B, D, H, W)
            # meta: (B, 3) -> [frag_idx, y, x]

            volumes = volumes.to(device, dtype=torch.float32)

            # Get TTA variants
            tta_variants = apply_tta(volumes)

            # Accumulate predictions for this batch across all TTA views
            batch_preds_sum = None

            for aug_vol, inverse_func in tta_variants:
                # Forward pass
                logits = model(aug_vol)
                probs = torch.sigmoid(logits)

                # Inverse transform to align with original orientation
                probs = inverse_func(probs)

                if batch_preds_sum is None:
                    batch_preds_sum = probs
                else:
                    batch_preds_sum += probs

            # Average over TTA variants
            batch_avg_preds = batch_preds_sum / len(tta_variants)

            # Move to CPU for accumulation
            batch_avg_preds = batch_avg_preds.cpu().numpy()
            meta = meta.numpy()

            # Accumulate into global maps
            for i in range(len(meta)):
                frag_idx, y, x = meta[i]

                # Extract the patch prediction (H, W)
                pred_patch = batch_avg_preds[i, 0]
                h_p, w_p = pred_patch.shape

                # Accumulate
                # y, x are top-left coordinates in the padded volume
                results[frag_idx]["prob"][y : y + h_p, x : x + w_p] += pred_patch
                results[frag_idx]["count"][y : y + h_p, x : x + w_p] += 1.0

    return results


def generate_submission(threshold=0.5, load_cached_data=True):
    """
    Main inference pipeline to generate submission.csv.

    Args:
        threshold (float): Probability threshold for binary classification.
        load_cached_data (bool): Whether to use cached dataset files.
    """
    print("--- Starting Inference Pipeline ---")

    # 1. Setup
    Config.set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Load Model
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train the model first."
        )

    print(f"Loading model from {model_path}...")
    model = PSDN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 3. Load Test Data
    print("Loading test dataset...")
    # InkDataset handles caching of volumes/masks internally
    test_dataset = InkDataset(split="test", transform=None, cache_data=load_cached_data)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Run Inference
    # Returns accumulated maps on padded arrays
    frag_results = predict_fragment(model, test_loader, device)

    # 5. Process and Format Output
    submission_data = []
    pad = Config.PATCH_SIZE // 2

    print(f"Processing results with threshold {threshold}...")

    # Iterate over fragments in the order they appear in metadata
    for idx, row in test_dataset.metadata.iterrows():
        frag_id = str(row["fragment_id"])

        # Retrieve accumulated maps
        res = frag_results[idx]
        prob_map = res["prob"]
        count_map = res["count"]

        # Normalize by count (handling division by zero for unvisited pixels)
        count_map[count_map == 0] = 1.0
        prob_map /= count_map

        # Crop padding to restore original dimensions
        # Original size from metadata
        h_orig = row["height"]
        w_orig = row["width"]

        # Crop center region (removing padding)
        prob_map_cropped = prob_map[pad : pad + h_orig, pad : pad + w_orig]

        # Apply Threshold
        binary_mask = (prob_map_cropped >= threshold).astype(np.uint8)

        # Apply Valid Mask
        # We use the mask from the dataset (which is padded), so we crop it similarly
        padded_mask = test_dataset.fragments[idx]["mask"]
        valid_mask = padded_mask[pad : pad + h_orig, pad : pad + w_orig]

        # Zero out predictions outside the valid fragment area
        binary_mask = binary_mask * (valid_mask > 0)

        # Run-Length Encoding
        rle = rle_encoding(binary_mask)

        submission_data.append({"Id": frag_id, "Predicted": rle})

        print(f"Fragment {frag_id}: Processed.")

    # 6. Save Submission
    df_sub = pd.DataFrame(submission_data)
    # Save to the home directory as required
    save_path = Config.SUBMISSION_FILE
    df_sub.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
