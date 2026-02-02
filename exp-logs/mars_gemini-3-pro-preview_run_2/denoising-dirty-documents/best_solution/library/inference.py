import os
import torch
import torch.nn.functional as F
import numpy as np
from library.config import Config
from library.model import CACResUNet
from library.dataset import TextDenoisingDataset
from library.utils import save_submission, seed_everything


def predict_tiled(model, image_tensor, patch_size, overlap=0.5, batch_size=4):
    """
    Performs inference on a large image using a sliding window (tiling) approach.

    Args:
        model: The trained PyTorch model.
        image_tensor: Input image tensor of shape (1, 1, H, W).
        patch_size: Size of the square tiles.
        overlap: Fraction of overlap between tiles (0.0 to 1.0).
        batch_size: Number of tiles to process in parallel.

    Returns:
        torch.Tensor: Reconstructed prediction tensor of shape (1, 1, H, W).
    """
    _, _, h, w = image_tensor.shape
    stride = int(patch_size * (1 - overlap))

    # Calculate padding needed to ensure the window covers the whole image
    # We pad the image so that the last window fits perfectly or extends slightly
    pad_h = (stride - (h - patch_size) % stride) % stride
    pad_w = (stride - (w - patch_size) % stride) % stride

    # Add extra padding if image is smaller than patch_size
    if h < patch_size:
        pad_h += patch_size - h
    if w < patch_size:
        pad_w += patch_size - w

    # Apply padding (Reflect padding is usually better for images)
    # Pad format: (left, right, top, bottom)
    padded_image = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, h_padded, w_padded = padded_image.shape

    # Initialize output and count containers
    output_sum = torch.zeros_like(padded_image)
    output_count = torch.zeros_like(padded_image)

    patches = []
    coords = []

    # Extract patches
    for y in range(0, h_padded - patch_size + 1, stride):
        for x in range(0, w_padded - patch_size + 1, stride):
            patch = padded_image[:, :, y : y + patch_size, x : x + patch_size]
            patches.append(patch)
            coords.append((y, x))

    # Process batches
    num_patches = len(patches)
    for i in range(0, num_patches, batch_size):
        batch_patches = torch.cat(patches[i : i + batch_size], dim=0)
        batch_coords = coords[i : i + batch_size]

        with torch.no_grad():
            # Model predicts noise residual
            batch_preds = model(batch_patches)

        # Accumulate predictions
        for pred, (y, x) in zip(batch_preds, batch_coords):
            # pred is (1, H, W)
            output_sum[:, :, y : y + patch_size, x : x + patch_size] += pred
            output_count[:, :, y : y + patch_size, x : x + patch_size] += 1.0

    # Average overlapping areas
    # Avoid division by zero (though logic ensures count >= 1)
    output = output_sum / output_count

    # Crop back to original size
    output = output[:, :, :h, :w]

    return output


def apply_tta(model, image_tensor, patch_size, overlap, device):
    """
    Applies Test-Time Augmentation (TTA) by averaging predictions
    from original and flipped versions of the input.
    """
    preds = []

    # 1. Original
    pred_orig = predict_tiled(model, image_tensor.to(device), patch_size, overlap)
    preds.append(pred_orig.cpu())

    # 2. Horizontal Flip
    img_hflip = torch.flip(image_tensor, [3])
    pred_hflip = predict_tiled(model, img_hflip.to(device), patch_size, overlap)
    preds.append(torch.flip(pred_hflip, [3]).cpu())

    # 3. Vertical Flip
    img_vflip = torch.flip(image_tensor, [2])
    pred_vflip = predict_tiled(model, img_vflip.to(device), patch_size, overlap)
    preds.append(torch.flip(pred_vflip, [2]).cpu())

    # Average predictions
    final_pred = torch.stack(preds).mean(dim=0)
    return final_pred


def run_inference(
    model_path=Config.MODEL_SAVE_PATH,
    output_path=Config.SUBMISSION_PATH,
    batch_size=Config.BATCH_SIZE,
    load_cached_data=True,
):
    """
    Main inference routine.
    Loads model, iterates over test set, performs TTA inference, and saves submission.
    """
    print(f"Starting inference using model: {model_path}")
    seed_everything()

    # Device
    device = Config.DEVICE

    # Load Model
    model = CACResUNet().to(device)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    else:
        print(
            f"Warning: Model file not found at {model_path}. Using random weights (Debug mode)."
        )

    model.eval()

    # Load Test Dataset
    # We use the dataset class to handle loading and caching logic
    test_dataset = TextDenoisingDataset(
        metadata_path=Config.TEST_METADATA,
        mode="test",
        load_cached_data=load_cached_data,
    )

    # Create DataLoader (batch_size=1 because images have different sizes)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=Config.NUM_WORKERS
    )

    predictions = {}

    print(f"Processing {len(test_dataset)} test images...")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # Unpack batch
            # Dataset returns: tensor_noisy (1, H, W), img_id (tuple/list)
            noisy_img, img_id_tuple = batch
            img_id = img_id_tuple[0]

            # Input is (B, 1, H, W) -> (1, 1, H, W)
            noisy_img = noisy_img.to(device)

            # Predict Noise Residual
            # Using Tiled Inference and TTA
            predicted_noise = apply_tta(
                model,
                noisy_img,
                patch_size=Config.PATCH_SIZE,
                overlap=Config.TILE_OVERLAP,
                device=device,
            )

            # Reconstruction: Clean = Noisy - Noise
            # Move back to CPU for subtraction and storage
            noisy_cpu = noisy_img.cpu()
            clean_pred = noisy_cpu - predicted_noise

            # Clamp to valid range [0, 1]
            clean_pred = torch.clamp(clean_pred, 0.0, 1.0)

            # Convert to numpy (H, W)
            clean_numpy = clean_pred.squeeze().numpy()

            predictions[img_id] = clean_numpy

            if (i + 1) % 5 == 0:
                print(f"Processed {i + 1}/{len(test_dataset)} images")

    # Save Submission
    print(f"Saving submission to {output_path}...")
    save_submission(predictions, output_path)
    print("Inference complete.")
