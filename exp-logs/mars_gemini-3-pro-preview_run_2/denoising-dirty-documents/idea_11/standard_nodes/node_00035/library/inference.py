import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.config import Config
from library.utils import set_seed, get_device
from library.model import CSKResUNet
from library.dataset import DenoisingDataset


def predict_tiled(model, image, device, tile_size=128, overlap=0.5):
    """
    Performs sliding window inference on a large image tensor.

    Args:
        model: The trained PyTorch model.
        image: Input image tensor of shape (1, 1, H, W).
        device: Computation device.
        tile_size: Size of the square patch.
        overlap: Overlap ratio (0.0 to 1.0).

    Returns:
        Tensor of shape (1, 1, H, W) containing the accumulated prediction.
    """
    _, _, H, W = image.shape
    stride = int(tile_size * (1 - overlap))

    # Accumulators for prediction and overlap counts
    output = torch.zeros((1, 1, H, W), device=device)
    count = torch.zeros((1, 1, H, W), device=device)

    # Calculate starting coordinates for tiles
    h_starts = list(range(0, H - tile_size + 1, stride))
    if H > tile_size and (len(h_starts) == 0 or h_starts[-1] != H - tile_size):
        h_starts.append(H - tile_size)
    if H <= tile_size:
        h_starts = [0]  # Fallback for small images (though unlikely based on EDA)

    w_starts = list(range(0, W - tile_size + 1, stride))
    if W > tile_size and (len(w_starts) == 0 or w_starts[-1] != W - tile_size):
        w_starts.append(W - tile_size)
    if W <= tile_size:
        w_starts = [0]

    # Sliding window loop
    for h_idx in h_starts:
        for w_idx in w_starts:
            # Extract patch
            # Handle case where image is smaller than tile_size by padding if necessary
            # (Though h_starts logic above aligns to borders for large images)
            if H < tile_size or W < tile_size:
                # This block handles the rare case of tiny images if they exist
                # For this dataset, images are large, so this is just a safeguard
                # We would need to pad input, predict, and crop.
                # Given dataset specs, we assume H, W >= tile_size usually.
                # If strictly needed, we'd implement padding here.
                pass

            # Standard slicing
            h_end = h_idx + tile_size
            w_end = w_idx + tile_size

            patch = image[:, :, h_idx:h_end, w_idx:w_end]

            with torch.no_grad():
                pred_patch = model(patch)

            output[:, :, h_idx:h_end, w_idx:w_end] += pred_patch
            count[:, :, h_idx:h_end, w_idx:w_end] += 1.0

    # Avoid division by zero
    count[count == 0] = 1.0

    return output / count


def apply_tta(model, image, device):
    """
    Applies Test-Time Augmentation (TTA) by predicting on geometric transformations
    and averaging the inverse-transformed results.
    """
    preds = []

    # 1. Original
    preds.append(
        predict_tiled(model, image, device, Config.TILE_SIZE, Config.TILE_OVERLAP)
    )

    if Config.USE_TTA:
        # 2. Horizontal Flip
        img_hf = torch.flip(image, [3])
        p_hf = predict_tiled(
            model, img_hf, device, Config.TILE_SIZE, Config.TILE_OVERLAP
        )
        preds.append(torch.flip(p_hf, [3]))

        # 3. Vertical Flip
        img_vf = torch.flip(image, [2])
        p_vf = predict_tiled(
            model, img_vf, device, Config.TILE_SIZE, Config.TILE_OVERLAP
        )
        preds.append(torch.flip(p_vf, [2]))

        # 4. Rotate 90 degrees (k=1)
        # Dimensions change (H, W) -> (W, H), handled by tiled inference
        img_r90 = torch.rot90(image, 1, [2, 3])
        p_r90 = predict_tiled(
            model, img_r90, device, Config.TILE_SIZE, Config.TILE_OVERLAP
        )
        preds.append(torch.rot90(p_r90, 3, [2, 3]))  # Inverse is rot90 k=3 (270 deg)

    # Average predictions
    return torch.stack(preds).mean(dim=0)


def run_inference(limit_size=None):
    """
    Main inference routine.
    Loads model, processes test set, and generates submission CSV.
    """
    set_seed(Config.SEED)
    device = get_device()
    print(f"Inference Device: {device}")

    # 1. Load Model
    print(f"Loading model from {Config.MODEL_SAVE_PATH}...")
    model = CSKResUNet().to(device)

    if os.path.exists(Config.MODEL_SAVE_PATH):
        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            "Warning: Model checkpoint not found. Using random weights (for debugging only)."
        )

    model.eval()

    # 2. Load Test Data
    print("Initializing test dataset...")
    test_dataset = DenoisingDataset(
        metadata_path=Config.TEST_METADATA,
        root_dir=Config.INPUT_DIR,
        mode="test",
        load_cached_data=True,
        limit_size=limit_size,
    )

    # Batch size 1 is mandatory for inference on varying image sizes
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    print(f"Processing {len(test_dataset)} test images...")

    # 3. Generate Predictions
    all_ids = []
    all_values = []

    with torch.no_grad():
        for noisy_img, img_id_tuple in test_loader:
            img_id = str(img_id_tuple[0])
            noisy_img = noisy_img.to(device)

            # Predict Noise Residual
            noise_pred = apply_tta(model, noisy_img, device)

            # Reconstruct Clean Image: Clean = Noisy - Noise
            clean_pred = noisy_img - noise_pred

            # Clamp to valid range [0, 1]
            clean_pred = torch.clamp(clean_pred, 0.0, 1.0)

            # Move to CPU and numpy
            clean_np = clean_pred.squeeze().cpu().numpy()  # (H, W)

            # Flatten and format for submission
            h, w = clean_np.shape

            # Generate IDs: {img_id}_{row}_{col} (1-based index)
            # Create grid of indices
            # Note: submission format is row_col, 1-based.
            # We iterate row by row.

            # Vectorized ID generation is tricky with strings, using list comprehension
            # Optimizing for speed:
            # Create row indices: [1, 1, ..., 2, 2, ...]
            # Create col indices: [1, 2, ..., 1, 2, ...]

            # Use list comprehension which is reasonably fast for ~200k pixels per image
            ids = [f"{img_id}_{r+1}_{c+1}" for r in range(h) for c in range(w)]
            values = clean_np.flatten().tolist()

            all_ids.extend(ids)
            all_values.extend(values)

    # 4. Create Submission File
    print("Constructing submission DataFrame...")
    df_submission = pd.DataFrame({"id": all_ids, "value": all_values})

    output_path = Config.SUBMISSION_PATH
    print(f"Saving submission to {output_path}...")
    df_submission.to_csv(output_path, index=False)
    print("Submission generation complete.")
