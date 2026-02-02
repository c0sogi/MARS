import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.model import ContrailUNet
from library.dataset import get_dataloader
from library.utils import rle_encode


def tta_inference(model, images):
    """
    Performs Test Time Augmentation (TTA) by averaging predictions
    from original, horizontally flipped, and vertically flipped images.

    Args:
        model (nn.Module): The trained neural network.
        images (torch.Tensor): Batch of input images (B, C, H, W).

    Returns:
        torch.Tensor: Averaged logits.
    """
    # 1. Original
    logits = model(images)

    # 2. Horizontal Flip
    images_hflip = torch.flip(images, dims=[3])
    logits_hflip = model(images_hflip)
    logits_hflip = torch.flip(logits_hflip, dims=[3])

    # 3. Vertical Flip
    images_vflip = torch.flip(images, dims=[2])
    logits_vflip = model(images_vflip)
    logits_vflip = torch.flip(logits_vflip, dims=[2])

    # Average predictions (logits)
    avg_logits = (logits + logits_hflip + logits_vflip) / 3.0

    return avg_logits


def make_submission(
    checkpoint_path,
    output_csv=Config.SUBMISSION_FILE,
    batch_size=Config.BATCH_SIZE,
    device=Config.DEVICE,
    threshold=0.5,
    debug=Config.DEBUG,
):
    """
    Generates predictions for the test set and saves them to a submission CSV.

    Args:
        checkpoint_path (str): Path to the trained model weights.
        output_csv (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        device (torch.device): Compute device.
        threshold (float): Probability threshold for binarization.
        debug (bool): If True, runs on a subset of the data.
    """
    # Initialize Model
    model = ContrailUNet()
    model.to(device)

    # Load Weights
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle state dict loading (check if wrapped in 'model_state_dict' or direct)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Get Test Loader
    test_loader = get_dataloader(
        split="test",
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        debug=debug,
    )

    results = []
    print(
        f"Starting inference on test set (TTA={'Enabled' if Config.USE_TTA else 'Disabled'})..."
    )

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device, non_blocking=True)
            record_ids = batch["record_id"]

            # Inference
            if Config.USE_TTA:
                logits = tta_inference(model, images)
            else:
                logits = model(images)

            # Convert logits to probabilities
            probs = torch.sigmoid(logits)

            # Binarize
            preds = (probs > threshold).float()

            # Move to CPU for encoding
            preds_np = preds.detach().cpu().numpy()  # Shape: (B, 1, H, W)

            # Iterate through batch
            for i, record_id in enumerate(record_ids):
                # Remove channel dim: (1, H, W) -> (H, W)
                mask = preds_np[i, 0, :, :]

                # RLE Encode
                encoded_string = rle_encode(mask)

                results.append(
                    {"record_id": record_id, "encoded_pixels": encoded_string}
                )

    # Create DataFrame
    submission_df = pd.DataFrame(results)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Save
    submission_df.to_csv(output_csv, index=False)
    print(f"Submission saved to {output_csv}")
