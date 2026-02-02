import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, rle_encode
from library.dataset import ContrailDataset, get_transforms
from library.model import ContextEnhancedResNet18UNet


def predict_tta(model, images, device):
    """
    Performs Test-Time Augmentation (TTA) by averaging predictions from:
    1. Original Image
    2. Horizontal Flip
    3. Vertical Flip
    4. 180-degree Rotation

    Args:
        model (nn.Module): The trained model.
        images (torch.Tensor): Batch of images (B, C, H, W).
        device (torch.device): Device to run inference on.

    Returns:
        torch.Tensor: Averaged probability maps (B, 1, H, W).
    """
    model.eval()
    with torch.no_grad():
        # 1. Original
        logits_orig = model(images)
        probs_orig = torch.sigmoid(logits_orig)

        # 2. Horizontal Flip
        images_h = torch.flip(images, dims=[3])
        logits_h = model(images_h)
        probs_h = torch.sigmoid(logits_h)
        probs_h = torch.flip(probs_h, dims=[3])

        # 3. Vertical Flip
        images_v = torch.flip(images, dims=[2])
        logits_v = model(images_v)
        probs_v = torch.sigmoid(logits_v)
        probs_v = torch.flip(probs_v, dims=[2])

        # 4. Rotate 180
        # k=2 rotates 90 degrees twice
        images_r = torch.rot90(images, k=2, dims=[2, 3])
        logits_r = model(images_r)
        probs_r = torch.sigmoid(logits_r)
        # Inverse rotation is the same for 180 degrees
        probs_r = torch.rot90(probs_r, k=2, dims=[2, 3])

        # Average probabilities
        avg_probs = (probs_orig + probs_h + probs_v + probs_r) / 4.0

    return avg_probs


def generate_submission(debug=False):
    """
    Generates the submission file for the test set.

    Args:
        debug (bool): If True, runs on a subset of data for debugging.
    """
    # Set seeds for reproducibility
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Running inference on device: {device}")

    # Initialize Model
    model = ContextEnhancedResNet18UNet(
        in_channels=Config.IN_CHANNELS, pretrained=False
    )
    model = model.to(device)

    # Load Best Weights
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.BEST_MODEL_PATH}"
        )

    print(f"Loading model weights from {Config.BEST_MODEL_PATH}...")
    checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Initialize Test Dataset and Loader
    test_dataset = ContrailDataset(
        split="test", transform=get_transforms("test", Config), debug=debug
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Starting inference on {len(test_dataset)} test images...")

    submission_data = []

    # Inference Loop
    for i, batch in enumerate(test_loader):
        images = batch["image"].to(device)
        record_ids = batch["record_id"]

        # Predict with TTA if enabled in Config, else standard predict
        if Config.USE_TTA:
            probs = predict_tta(model, images, device)
        else:
            with torch.no_grad():
                logits = model(images)
                probs = torch.sigmoid(logits)

        # Move to CPU and convert to numpy for RLE encoding
        probs_np = probs.cpu().numpy()

        # Apply threshold and encode
        for j in range(len(record_ids)):
            # Extract single mask: (1, H, W) -> (H, W)
            pred_mask = probs_np[j, 0, :, :]

            # Binarize
            binary_mask = (pred_mask > Config.THRESHOLD).astype(np.uint8)

            # Encode
            rle_str = rle_encode(binary_mask)

            submission_data.append(
                {"record_id": record_ids[j], "encoded_pixels": rle_str}
            )

        if debug and i >= 5:
            print("Debug mode: stopping after 5 batches.")
            break

    # Create DataFrame
    submission_df = pd.DataFrame(submission_data)

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save to CSV
    print(f"Saving submission to {Config.SUBMISSION_FILE}...")
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)

    print("Submission generation completed successfully.")
