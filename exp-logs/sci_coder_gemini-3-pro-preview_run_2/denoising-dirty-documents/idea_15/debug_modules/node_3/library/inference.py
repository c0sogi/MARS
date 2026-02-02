import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.config import Config
from library.utils import set_seed, create_submission, load_image, normalize_image
from library.model import RepCResUNetSR


def predict_tiled(
    model, image_tensor, patch_size=128, overlap=0.5, batch_size=32, device="cuda"
):
    """
    Performs tiled inference on a large image with overlapping patches.

    Args:
        model (nn.Module): The trained PyTorch model (in deploy mode).
        image_tensor (torch.Tensor): Input image tensor of shape (1, 1, H, W).
        patch_size (int): Size of the square patch.
        overlap (float): Overlap ratio between patches (0.0 to 1.0).
        batch_size (int): Number of patches to process in a single batch.
        device (str): Device to run inference on.

    Returns:
        torch.Tensor: Predicted image tensor of shape (1, 1, H, W).
    """
    _, _, h, w = image_tensor.shape
    stride = int(patch_size * (1 - overlap))

    # Calculate padding to ensure the sliding window covers the entire image
    # We pad the right and bottom edges
    pad_h = 0
    pad_w = 0

    if h < patch_size:
        pad_h = patch_size - h
    elif (h - patch_size) % stride != 0:
        pad_h = stride - ((h - patch_size) % stride)

    if w < patch_size:
        pad_w = patch_size - w
    elif (w - patch_size) % stride != 0:
        pad_w = stride - ((w - patch_size) % stride)

    # Apply padding using reflection to maintain texture continuity at borders
    # F.pad expects (left, right, top, bottom)
    padded_image = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode="reflect")

    _, _, h_pad, w_pad = padded_image.shape

    # Tensors to accumulate predictions and count overlaps
    output_sum = torch.zeros_like(padded_image)
    output_count = torch.zeros_like(padded_image)

    patches = []
    coords = []

    # Extract patches
    for y in range(0, h_pad - patch_size + 1, stride):
        for x in range(0, w_pad - patch_size + 1, stride):
            patch = padded_image[:, :, y : y + patch_size, x : x + patch_size]
            patches.append(patch)
            coords.append((y, x))

    if not patches:
        return image_tensor

    # Stack patches into a batch: (N, 1, P, P)
    patches_tensor = torch.cat(patches, dim=0)

    predicted_patches = []

    # Run inference in batches
    with torch.no_grad():
        for i in range(0, len(patches_tensor), batch_size):
            batch = patches_tensor[i : i + batch_size].to(device)
            pred = model(batch)
            predicted_patches.append(pred.cpu())

    predicted_patches = torch.cat(predicted_patches, dim=0)

    # Reconstruct the image
    for i, (y, x) in enumerate(coords):
        output_sum[:, :, y : y + patch_size, x : x + patch_size] += predicted_patches[
            i : i + 1
        ].to(output_sum.device)
        output_count[:, :, y : y + patch_size, x : x + patch_size] += 1.0

    # Average overlapping regions
    output = output_sum / output_count

    # Crop back to original dimensions
    output = output[:, :, :h, :w]

    return output


def predict_tta(model, image_tensor, patch_size=128, overlap=0.5, device="cuda"):
    """
    Performs Test-Time Augmentation (TTA) by predicting on geometric transformations
    and averaging the results.

    Transforms applied:
    1. Original
    2. Horizontal Flip
    3. Vertical Flip
    4. Rotate 180 (equivalent to HFlip + VFlip)

    Args:
        model (nn.Module): Trained model.
        image_tensor (torch.Tensor): Input image (1, 1, H, W).
        patch_size (int): Patch size for tiled inference.
        overlap (float): Overlap ratio.
        device (str): Device.

    Returns:
        torch.Tensor: Averaged prediction.
    """
    preds = []

    # 1. Original
    p1 = predict_tiled(model, image_tensor, patch_size, overlap, device=device)
    preds.append(p1)

    # 2. Horizontal Flip
    img_h = torch.flip(image_tensor, [3])
    p2 = predict_tiled(model, img_h, patch_size, overlap, device=device)
    p2 = torch.flip(p2, [3])  # Inverse transform
    preds.append(p2)

    # 3. Vertical Flip
    img_v = torch.flip(image_tensor, [2])
    p3 = predict_tiled(model, img_v, patch_size, overlap, device=device)
    p3 = torch.flip(p3, [2])  # Inverse transform
    preds.append(p3)

    # 4. Rotate 180
    img_180 = torch.rot90(image_tensor, 2, [2, 3])
    p4 = predict_tiled(model, img_180, patch_size, overlap, device=device)
    p4 = torch.rot90(p4, -2, [2, 3])  # Inverse transform
    preds.append(p4)

    # Average predictions
    final_pred = torch.stack(preds).mean(dim=0)
    return final_pred


def generate_submission(
    checkpoint_path=Config.BEST_MODEL_PATH,
    output_path=Config.SUBMISSION_PATH,
    test_csv_path=Config.TEST_CSV,
    limit=None,
    device=Config.DEVICE,
):
    """
    Generates the submission file for the test dataset.

    Args:
        checkpoint_path (str): Path to the trained model checkpoint.
        output_path (str): Path to save the submission CSV.
        test_csv_path (str): Path to the test metadata CSV.
        limit (int, optional): Limit the number of images processed (for debugging).
        device (str): Device to run inference on.
    """
    set_seed(Config.SEED)

    # 1. Load Model
    print(f"Loading model from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model = RepCResUNetSR()
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2. Switch to Deploy Mode
    # This fuses the multi-branch RepBlocks into single 3x3 convolutions
    print("Switching model to deploy mode (fusing reparameterized blocks)...")
    model.switch_to_deploy()

    # 3. Load Test Metadata
    df_test = pd.read_csv(test_csv_path)
    if limit is not None:
        df_test = df_test.head(limit)
    print(f"Processing {len(df_test)} test images...")

    predictions = {}

    # 4. Inference Loop
    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Inference"):
        img_id = str(row["id"])
        rel_path = row["feature_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load and Preprocess
        # Attempt to load from cache if available, otherwise load from source
        cache_path = os.path.join(Config.CACHE_DIR, rel_path.replace(".png", ".npy"))

        try:
            img_arr = load_image(full_path, cache_path=cache_path)
        except Exception as e:
            print(f"Error loading image {img_id}: {e}")
            continue

        img_arr = normalize_image(img_arr)

        # Convert to Tensor: (H, W) -> (1, 1, H, W)
        img_tensor = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0).to(device)

        # Run TTA Inference
        # Using patch size 128 and 50% overlap as per strategy
        pred_tensor = predict_tta(
            model, img_tensor, patch_size=Config.PATCH_SIZE, overlap=0.5, device=device
        )

        # Post-process
        pred_arr = pred_tensor.squeeze().cpu().numpy()

        # Clip values to valid range [0, 1]
        pred_arr = np.clip(pred_arr, 0.0, 1.0)

        predictions[img_id] = pred_arr

    # 5. Create Submission File
    print(f"Generating submission file at {output_path}...")
    create_submission(predictions, output_path)
    print("Submission generation complete.")
