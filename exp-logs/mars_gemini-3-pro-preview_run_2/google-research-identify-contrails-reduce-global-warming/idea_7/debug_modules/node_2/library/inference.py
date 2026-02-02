import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import ContrailDataset, get_valid_transform
from library.model import DeformableResNetUNet
from library.utils import rle_encode, seed_everything


def predict_with_tta(model, images):
    """
    Performs Test-Time Augmentation (TTA) by averaging predictions across
    original, horizontally flipped, vertically flipped, and 180-degree rotated views.

    Args:
        model (nn.Module): The trained model.
        images (torch.Tensor): Batch of images (B, C, H, W).

    Returns:
        torch.Tensor: Averaged probability maps (B, 1, H, W).
    """
    # 1. Original
    out = torch.sigmoid(model(images))

    # 2. Horizontal Flip
    # Flip input dims=[3] (width)
    img_h = torch.flip(images, dims=[3])
    out_h = torch.sigmoid(model(img_h))
    # Flip output back
    out += torch.flip(out_h, dims=[3])

    # 3. Vertical Flip
    # Flip input dims=[2] (height)
    img_v = torch.flip(images, dims=[2])
    out_v = torch.sigmoid(model(img_v))
    # Flip output back
    out += torch.flip(out_v, dims=[2])

    # 4. Rotate 180 (Horizontal + Vertical Flip)
    img_hv = torch.flip(images, dims=[2, 3])
    out_hv = torch.sigmoid(model(img_hv))
    # Flip output back
    out += torch.flip(out_hv, dims=[2, 3])

    # Average
    return out / 4.0


def run_inference(
    checkpoint_path=Config.BEST_MODEL_PATH,
    output_path=Config.SUBMISSION_PATH,
    batch_size=Config.BATCH_SIZE,
    device=Config.DEVICE,
    threshold=Config.THRESHOLD,
    debug=Config.DEBUG,
):
    """
    Main inference function. Loads model, generates predictions on test set,
    and saves submission file.

    Args:
        checkpoint_path (str): Path to the saved model weights.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        device (str): Device to run inference on.
        threshold (float): Threshold for binarizing probabilities.
        debug (bool): If True, runs on a subset of data.
    """
    # 1. Reproducibility
    seed_everything(Config.SEED)

    # 2. Load Metadata
    if not os.path.exists(Config.TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"Test metadata not found at {Config.TEST_METADATA_PATH}"
        )

    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # 3. Setup Dataset and DataLoader
    # We need return_record_id=True to map predictions to IDs
    dataset = ContrailDataset(
        test_df, transform=get_valid_transform(), debug=debug, return_record_id=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Initialize Model
    model = DeformableResNetUNet(
        n_channels=Config.N_CHANNELS,
        n_classes=1,
        pretrained=False,  # No need to download weights, we load checkpoint
    )

    # Load weights
    if not os.path.exists(checkpoint_path):
        print(
            f"Warning: Checkpoint {checkpoint_path} not found. Using random weights (for testing flow only)."
        )
    else:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    results = []

    # 5. Inference Loop
    print(f"Starting inference on {len(dataset)} samples...")

    with torch.no_grad():
        for images, _, record_ids in dataloader:
            images = images.to(device, non_blocking=True)

            # Apply TTA
            # Output shape: (B, 1, H, W)
            probs = predict_with_tta(model, images)

            # Move to CPU for post-processing
            probs = probs.cpu().numpy()

            # Process batch
            for i in range(len(record_ids)):
                # Extract single mask: (1, H, W) -> (H, W)
                prob_map = probs[i, 0, :, :]

                # Threshold
                binary_mask = (prob_map > threshold).astype(np.uint8)

                # RLE Encode
                rle_str = rle_encode(binary_mask)

                # Handle empty predictions
                if rle_str == "":
                    rle_str = "-"

                results.append({"record_id": record_ids[i], "encoded_pixels": rle_str})

    # 6. Save Submission
    submission_df = pd.DataFrame(results)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} with {len(submission_df)} records.")
