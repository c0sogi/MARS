import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import ContrailDataset
from library.model import ResnetUNet
from library.utils import rle_encode


def predict(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    device=Config.DEVICE,
    threshold=Config.THRESHOLD,
):
    """
    Runs inference on the test dataset and generates the submission file.

    Args:
        batch_size (int): Batch size for inference.
        num_workers (int): Number of workers for data loading.
        device (torch.device): Device to run inference on.
        threshold (float): Threshold for binarizing probability masks.
    """
    # Set seeds for reproducibility
    Config.set_seed(Config.SEED)

    print(f"Starting inference on device: {device}")

    # 1. Load Model
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please train the model first."
        )

    print(f"Loading model from {model_path}...")
    model = ResnetUNet(in_channels=Config.IN_CHANNELS, pretrained=False)

    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # 2. Prepare Data
    print("Initializing Test Dataset...")
    test_dataset = ContrailDataset(split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Total test samples: {len(test_dataset)}")

    # 3. Inference Loop
    submission_data = []

    with torch.no_grad():
        for i, (images, _, record_ids) in enumerate(test_loader):
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            pred_masks = outputs

            # Apply threshold to get binary mask
            # pred_masks is [B, 1, H, W]
            binary_masks = (pred_masks > threshold).float()

            # Move to CPU for RLE encoding
            binary_masks_np = binary_masks.cpu().numpy()

            # Iterate over batch
            for j in range(len(record_ids)):
                record_id = record_ids[j]

                # Extract single mask: [1, H, W] -> [H, W]
                mask = binary_masks_np[j, 0, :, :]

                # Encode
                rle_str = rle_encode(mask)

                submission_data.append(
                    {"record_id": record_id, "encoded_pixels": rle_str}
                )

    # 4. Save Submission
    print("Generating submission file...")
    df_submission = pd.DataFrame(submission_data)

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    output_path = Config.SUBMISSION_FILE
    df_submission.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
    print(df_submission.head())
