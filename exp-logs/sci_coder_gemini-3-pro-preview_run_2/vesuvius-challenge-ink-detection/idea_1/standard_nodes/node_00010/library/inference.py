import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from library.config import Config
from library.utils import seed_everything, rle_encode
from library.architecture import MIPUNet
from library.dataset import get_fragment_mip, get_transforms


def predict_fragment(
    fragment_id,
    vol_path,
    mask_path,
    model,
    device,
    load_cached_data=True,
    batch_size=Config.BATCH_SIZE,
):
    """
    Generates a probability map for a single fragment using tiled inference.

    Args:
        fragment_id (str): The ID of the fragment.
        vol_path (str): Relative path to the volume directory.
        mask_path (str): Relative path to the binary mask.
        model (torch.nn.Module): The loaded model.
        device (torch.device): Computation device.
        load_cached_data (bool): Whether to use cached MIPs.
        batch_size (int): Batch size for inference.

    Returns:
        np.ndarray: Probability map of the fragment (0.0 to 1.0).
    """
    # 1. Load Data
    # Load MIP (Cached or Computed)
    mip = get_fragment_mip(fragment_id, vol_path, load_cached_data=load_cached_data)

    # Load Mask (to define valid area)
    mask_full_path = os.path.join(Config.INPUT_DIR, mask_path)
    if not os.path.exists(mask_full_path):
        raise FileNotFoundError(f"Mask file not found: {mask_full_path}")

    mask_img = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise ValueError(f"Failed to load mask image: {mask_full_path}")

    valid_mask = (mask_img > 0).astype(bool)

    # 2. Preprocess
    # Normalize to [0, 1] float32 (same as training)
    # Original data is uint16
    mip = mip.astype(np.float32) / 65535.0

    h, w = mip.shape[:2]
    tile_size = Config.TILE_SIZE

    # Pad image to be divisible by TILE_SIZE
    pad_h = (tile_size - (h % tile_size)) % tile_size
    pad_w = (tile_size - (w % tile_size)) % tile_size

    if mip.ndim == 3:
        mip_padded = np.pad(
            mip, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0
        )
    else:
        mip_padded = np.pad(
            mip, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
        )

    # 3. Tiled Inference preparation
    transforms = get_transforms("test")
    patches = []
    coords = []

    # Extract tiles
    for y in range(0, mip_padded.shape[0], tile_size):
        for x in range(0, mip_padded.shape[1], tile_size):
            patch = mip_padded[y : y + tile_size, x : x + tile_size]

            # Apply transforms (Normalize + ToTensor)
            augmented = transforms(image=patch)
            patch_tensor = augmented["image"]

            patches.append(patch_tensor)
            coords.append((y, x))

    # Create DataLoader for batch processing
    # Stack patches into a single tensor: (N, C, H, W)
    patches_tensor = torch.stack(patches)
    dataset = TensorDataset(patches_tensor)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=Config.NUM_WORKERS
    )

    # 4. Run Inference
    preds_padded = np.zeros(mip_padded.shape[:2], dtype=np.float32)

    model.eval()
    patch_idx = 0

    with torch.no_grad():
        for batch in loader:
            batch_imgs = batch[0].to(device)
            outputs = model(batch_imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()

            # Place predictions back into the canvas
            for i in range(probs.shape[0]):
                py, px = coords[patch_idx]
                # probs shape is (B, 1, H, W), take channel 0
                preds_padded[py : py + tile_size, px : px + tile_size] = probs[i, 0]
                patch_idx += 1

    # 5. Post-process
    # Crop back to original size
    preds = preds_padded[:h, :w]

    # Apply valid mask (zero out predictions outside the fragment)
    preds = preds * valid_mask

    return preds


def create_submission(
    model_path,
    submission_output_path=Config.SUBMISSION_PATH,
    test_metadata_path=Config.TEST_METADATA_PATH,
    threshold=0.5,
    load_cached_data=True,
):
    """
    Runs the full inference pipeline and generates the submission CSV.

    Args:
        model_path (str): Path to the trained model weights (.pth).
        submission_output_path (str): Path to save the submission CSV.
        test_metadata_path (str): Path to the test metadata CSV.
        threshold (float): Threshold to convert probabilities to binary mask.
        load_cached_data (bool): Whether to use cached MIPs.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Check paths
    if not os.path.exists(test_metadata_path):
        print(f"Test metadata not found at {test_metadata_path}. Skipping inference.")
        return

    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Cannot run inference.")
        return

    # --- Load Model ---
    print(f"Loading model from {model_path}...")
    model = MIPUNet(
        encoder_name=Config.ENCODER_NAME,
        encoder_weights=None,  # Loading custom weights
        in_channels=Config.IN_CHANNELS,
        classes=Config.CLASSES,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- Load Metadata ---
    test_df = pd.read_csv(test_metadata_path)
    submission_data = []

    print(f"Starting inference on {len(test_df)} fragments...")

    for _, row in test_df.iterrows():
        frag_id = row["fragment_id"]
        vol_path = row["volume_path"]
        mask_path = row["mask_path"]

        print(f"Processing fragment {frag_id}...")

        try:
            # Predict
            preds = predict_fragment(
                fragment_id=frag_id,
                vol_path=vol_path,
                mask_path=mask_path,
                model=model,
                device=device,
                load_cached_data=load_cached_data,
                batch_size=Config.BATCH_SIZE,
            )

            # Threshold to binary
            binary_preds = (preds > threshold).astype(np.uint8)

            # RLE Encode
            rle = rle_encode(binary_preds)
            submission_data.append({"Id": frag_id, "Predicted": rle})

        except Exception as e:
            print(f"Error processing fragment {frag_id}: {e}")
            # Append empty prediction in case of error to maintain row count if needed,
            # though ideally we fix the error.
            submission_data.append({"Id": frag_id, "Predicted": ""})

    # --- Save Submission ---
    sub_df = pd.DataFrame(submission_data)

    # Ensure output directory exists (though usually it's root)
    output_dir = os.path.dirname(submission_output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    sub_df.to_csv(submission_output_path, index=False)
    print(f"Submission saved to {submission_output_path}")
