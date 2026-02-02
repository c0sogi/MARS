import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import ContrailDataset
from library.model import ResNetUNet
from library.utils import rle_encode, seed_everything


def generate_submission(model_path, device=None, batch_size=None):
    """
    Generates the submission file for the test dataset using the trained model.

    Args:
        model_path (str): Path to the trained model weights (.pth file).
        device (str, optional): Device to run inference on ('cpu' or 'cuda').
                                Defaults to Config.DEVICE.
        batch_size (int, optional): Batch size for inference. Defaults to Config.BATCH_SIZE.
    """
    # 1. Setup Configuration
    if device is None:
        device = Config.DEVICE

    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    seed_everything(Config.SEED)

    print(f"Starting inference pipeline on device: {device}")

    # 2. Load Test Metadata
    if not os.path.exists(Config.TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"Test metadata not found at {Config.TEST_METADATA_PATH}"
        )

    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    # Ensure record_id is treated as string to match submission format
    test_df["record_id"] = test_df["record_id"].astype(str)

    print(f"Loaded {len(test_df)} test records.")

    # 3. Initialize Dataset and DataLoader
    # The ContrailDataset with split='test' will return only images (no masks)
    test_dataset = ContrailDataset(test_df, split="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Must be False to maintain alignment with record_ids
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # 4. Initialize Model and Load Weights
    model = ResNetUNet(in_channels=Config.IN_CHANNELS, num_classes=1)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights file not found at {model_path}")

    print(f"Loading model weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # 5. Inference Loop
    encoded_pixels = []

    print("Running prediction loop...")

    with torch.no_grad():
        for i, images in enumerate(test_loader):
            images = images.to(device, dtype=torch.float32)

            # Forward pass
            logits = model(images)

            # Apply sigmoid activation
            probs = torch.sigmoid(logits)

            # Threshold probabilities to get binary mask
            preds = (probs > Config.THRESHOLD).float()

            # Move predictions to CPU and convert to numpy
            preds_np = preds.detach().cpu().numpy()

            # Process each image in the batch
            # preds_np shape: (Batch_Size, 1, Height, Width)
            for j in range(preds_np.shape[0]):
                # Extract the single channel mask: (Height, Width)
                mask = preds_np[j, 0, :, :]

                # Perform Run-Length Encoding
                rle = rle_encode(mask)
                encoded_pixels.append(rle)

    # 6. Construct Submission DataFrame
    # Since shuffle=False, the order of encoded_pixels matches the order of test_df
    if len(encoded_pixels) != len(test_df):
        raise RuntimeError(
            f"Mismatch in predictions count: {len(encoded_pixels)} vs {len(test_df)}"
        )

    submission_df = pd.DataFrame(
        {"record_id": test_df["record_id"], "encoded_pixels": encoded_pixels}
    )

    # 7. Save Submission File
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission successfully saved to {Config.SUBMISSION_PATH}")
