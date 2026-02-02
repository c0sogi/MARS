import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import get_logger, load_checkpoint
from library.model import CoConvNeXtUNet

logger = get_logger("InferenceModule")


def predict_full_image(model, image, patch_size, overlap_ratio, device):
    """
    Predicts a full image using a sliding window approach with overlap.

    Args:
        model (nn.Module): The trained model.
        image (torch.Tensor): Input image tensor of shape (C, H, W).
        patch_size (int): Size of the square patch.
        overlap_ratio (float): Ratio of overlap between patches (0 to 1).
        device (torch.device): Computation device.

    Returns:
        torch.Tensor: Predicted clean image tensor of shape (C, H, W).
    """
    c, h, w = image.shape
    stride = int(patch_size * (1 - overlap_ratio))

    # Calculate padding required
    pad_h = (patch_size - h % stride) % stride
    pad_w = (patch_size - w % stride) % stride

    # Ensure dimensions are at least patch_size
    if h < patch_size:
        pad_h += patch_size - h
    if w < patch_size:
        pad_w += patch_size - w

    # Pad image (Reflect padding reduces boundary artifacts)
    padded_image = F.pad(
        image.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect"
    ).squeeze(0)
    ph, pw = padded_image.shape[1], padded_image.shape[2]

    output_sum = torch.zeros_like(padded_image)
    output_count = torch.zeros_like(padded_image)

    patches = []
    coords = []

    # Extract patches
    for y in range(0, ph - patch_size + 1, stride):
        for x in range(0, pw - patch_size + 1, stride):
            patch = padded_image[:, y : y + patch_size, x : x + patch_size]
            patches.append(patch)
            coords.append((y, x))

    # Process batches
    batch_size = Config.BATCH_SIZE

    # Ensure model is in eval mode
    model.eval()

    for i in range(0, len(patches), batch_size):
        batch_patches = patches[i : i + batch_size]
        batch_tensor = torch.stack(batch_patches).to(device)

        with torch.no_grad():
            # Model predicts noise residual
            pred_noise = model(batch_tensor)

            # Clean = Noisy - Noise
            pred_clean_batch = batch_tensor - pred_noise
            pred_clean_batch = torch.clamp(pred_clean_batch, 0, 1)

        # Accumulate results
        for j, pred in enumerate(pred_clean_batch):
            y, x = coords[i + j]
            output_sum[:, y : y + patch_size, x : x + patch_size] += pred.cpu()
            output_count[:, y : y + patch_size, x : x + patch_size] += 1.0

    # Average overlapping areas
    output = output_sum / output_count

    # Crop back to original size
    return output[:, :h, :w]


def apply_tta(model, image, patch_size, overlap_ratio, device):
    """
    Applies Test-Time Augmentation (TTA) by averaging predictions from
    geometric transformations.

    Args:
        model (nn.Module): The trained model.
        image (torch.Tensor): Input image tensor (C, H, W).
        patch_size (int): Patch size.
        overlap_ratio (float): Overlap ratio.
        device (torch.device): Device.

    Returns:
        torch.Tensor: Averaged prediction.
    """
    # Define transforms: (Forward, Inverse)
    transforms = [
        # Identity
        (lambda x: x, lambda x: x),
        # Horizontal Flip
        (lambda x: torch.flip(x, [2]), lambda x: torch.flip(x, [2])),
        # Vertical Flip
        (lambda x: torch.flip(x, [1]), lambda x: torch.flip(x, [1])),
        # Rotate 90 degrees (k=1)
        (lambda x: torch.rot90(x, 1, [1, 2]), lambda x: torch.rot90(x, -1, [1, 2])),
    ]

    ensemble_pred = None

    for fwd, inv in transforms:
        # Apply forward transform
        aug_img = fwd(image)

        # Predict
        pred = predict_full_image(model, aug_img, patch_size, overlap_ratio, device)

        # Apply inverse transform
        pred = inv(pred)

        if ensemble_pred is None:
            ensemble_pred = pred
        else:
            ensemble_pred += pred

    # Average
    return ensemble_pred / len(transforms)


def generate_submission(test_loader, device):
    """
    Generates the submission file for the test dataset.

    Args:
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Computation device.
    """
    logger.info("Starting submission generation...")

    # Initialize Model
    model = CoConvNeXtUNet(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        base_filters=Config.BASE_FILTERS,
    ).to(device)

    # Load Best Weights
    try:
        load_checkpoint(Config.MODEL_PATH, model, device=device)
        logger.info(f"Loaded model weights from {Config.MODEL_PATH}")
    except FileNotFoundError:
        logger.error(
            f"Model checkpoint not found at {Config.MODEL_PATH}. Cannot generate submission."
        )
        return

    model.eval()
    results = []

    # Process Test Images
    with torch.no_grad():
        for i, (noisy_batch, img_ids) in enumerate(test_loader):
            # Batch size is likely 1 for test loader to handle varying image sizes easily,
            # but logic supports batching if images were same size.
            # Here we assume batch size 1 or iterate through batch.

            for j in range(len(noisy_batch)):
                noisy_img = noisy_batch[j]  # (C, H, W)
                img_id = img_ids[j]

                # Apply TTA Inference
                if Config.TTA_ENABLED:
                    pred_clean = apply_tta(
                        model,
                        noisy_img,
                        Config.PATCH_SIZE,
                        Config.OVERLAP_RATIO,
                        device,
                    )
                else:
                    pred_clean = predict_full_image(
                        model,
                        noisy_img,
                        Config.PATCH_SIZE,
                        Config.OVERLAP_RATIO,
                        device,
                    )

                # Flatten for CSV format
                # pred_clean is (C, H, W). C=1.
                vals = pred_clean.squeeze(0).numpy().flatten()

                h, w = pred_clean.shape[1], pred_clean.shape[2]

                # Generate IDs: {img_id}_{row}_{col}
                # np.indices returns grid of indices
                grid_y, grid_x = np.indices((h, w))

                # Flatten indices (1-based indexing as per requirement)
                rows = grid_y.flatten() + 1
                cols = grid_x.flatten() + 1

                # Create ID strings efficiently
                # Using list comprehension is generally fast enough for this scale
                ids = [f"{img_id}_{r}_{c}" for r, c in zip(rows, cols)]

                # Append to results
                df_chunk = pd.DataFrame({"id": ids, "value": vals})
                results.append(df_chunk)

            if (i + 1) % 5 == 0:
                logger.info(f"Processed {i + 1} batches.")

    # Concatenate all results
    if results:
        final_df = pd.concat(results, ignore_index=True)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        # Save
        final_df.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
        logger.info(f"Total rows: {len(final_df)}")
    else:
        logger.warning("No results generated.")
