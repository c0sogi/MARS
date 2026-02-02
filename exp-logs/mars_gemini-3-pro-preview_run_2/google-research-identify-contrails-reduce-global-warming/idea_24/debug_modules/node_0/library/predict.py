import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import rle_encode
from library.dataset import ContrailDataset, get_transforms
from library.model import ProgressiveConvNeXtUNet


def predict_tta(model, images, device):
    """
    Performs Test-Time Augmentation (TTA) by averaging predictions from:
    1. Original
    2. Horizontal Flip
    3. Vertical Flip
    4. 180-degree Rotation (H-Flip + V-Flip)

    Args:
        model (nn.Module): The trained segmentation model.
        images (torch.Tensor): Batch of input images (B, C, H, W).
        device (torch.device): Device to perform computation on.

    Returns:
        torch.Tensor: Averaged probability maps (B, 1, H, W).
    """
    model.eval()
    with torch.no_grad():
        # 1. Original
        logits_orig = model(images)
        probs_orig = torch.sigmoid(logits_orig)

        # 2. Horizontal Flip (Flip on width dim, usually last dim 3)
        images_h = torch.flip(images, dims=[3])
        logits_h = model(images_h)
        probs_h = torch.sigmoid(logits_h)
        probs_h = torch.flip(probs_h, dims=[3])  # Flip back

        # 3. Vertical Flip (Flip on height dim, usually dim 2)
        images_v = torch.flip(images, dims=[2])
        logits_v = model(images_v)
        probs_v = torch.sigmoid(logits_v)
        probs_v = torch.flip(probs_v, dims=[2])  # Flip back

        # 4. 180 Rotation (Flip on both dims 2 and 3)
        images_rot = torch.flip(images, dims=[2, 3])
        logits_rot = model(images_rot)
        probs_rot = torch.sigmoid(logits_rot)
        probs_rot = torch.flip(probs_rot, dims=[2, 3])  # Flip back

        # Average probabilities
        avg_probs = (probs_orig + probs_h + probs_v + probs_rot) / 4.0

    return avg_probs


def generate_submission(
    checkpoint_path=Config.BEST_MODEL_PATH,
    batch_size=Config.BATCH_SIZE,
    output_path=Config.SUBMISSION_PATH,
    device_name=Config.DEVICE,
):
    """
    Generates the submission file for the test set.

    Args:
        checkpoint_path (str): Path to the trained model weights.
        batch_size (int): Batch size for inference.
        output_path (str): Path to save the submission CSV.
        device_name (str): Device to run inference on ('cuda' or 'cpu').
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    device = torch.device(device_name)
    print(f"Running inference on device: {device}")

    # 1. Load Data
    # Use pre-generated test metadata
    if not os.path.exists(Config.TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"Test metadata not found at {Config.TEST_METADATA_PATH}"
        )

    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Initialize Dataset and Loader
    # Use validation transforms (normalization only) for test set
    test_dataset = ContrailDataset(
        test_df, split="test", transform=get_transforms("validation")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Load Model
    model = ProgressiveConvNeXtUNet()

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. Inference Loop
    results = []

    print(f"Starting inference on {len(test_dataset)} images...")

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            record_ids = batch["record_id"]

            # Predict with TTA
            if Config.USE_TTA:
                probs = predict_tta(model, images, device)
            else:
                logits = model(images)
                probs = torch.sigmoid(logits)

            # Thresholding
            preds = (probs > Config.THRESHOLD).float()

            # Convert to numpy for RLE encoding
            preds_np = preds.cpu().numpy()

            # Iterate over batch
            for i in range(len(record_ids)):
                rid = record_ids[i]
                # Shape is (1, H, W), flatten to (H, W)
                mask = preds_np[i, 0, :, :]

                # Encode
                rle_str = rle_encode(mask)

                results.append({"record_id": rid, "encoded_pixels": rle_str})

    # 4. Save Submission
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
    print(f"Total records processed: {len(submission_df)}")
