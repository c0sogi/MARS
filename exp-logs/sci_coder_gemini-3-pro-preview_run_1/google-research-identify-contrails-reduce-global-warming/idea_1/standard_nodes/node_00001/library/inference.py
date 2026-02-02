import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import ContrailDataset
from library.model import SimpleUNet
from library.utils import rle_encode


def generate_submission(
    checkpoint_path=None,
    batch_size=Config.BATCH_SIZE,
    device=Config.DEVICE,
    threshold=Config.THRESHOLD,
    max_samples=None,
):
    """
    Generates the submission file for the test set using the trained model.

    Args:
        checkpoint_path (str, optional): Path to the model checkpoint.
                                         Defaults to Config.CHECKPOINT_DIR/best_model.pth.
        batch_size (int): Batch size for inference.
        device (str): Device to run inference on ('cpu' or 'cuda').
        threshold (float): Threshold to convert probabilities to binary mask.
        max_samples (int, optional): Limit number of samples for debugging.
    """

    # 1. Setup Paths and Device
    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Generating submission using model: {checkpoint_path}")
    print(f"Device: {device}")

    # 2. Load Data
    # We use the test split. The dataset internally loads metadata/test.csv.
    test_dataset = ContrailDataset(split="test", max_samples=max_samples)

    # Shuffle must be False to align with the dataset.df indices
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Test samples: {len(test_dataset)}")

    # 3. Load Model
    model = SimpleUNet(in_channels=Config.NUM_BANDS, out_channels=1)

    # Load weights
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading state dict: {e}")
        return

    model.to(device)
    model.eval()

    # 4. Inference Loop
    results = []

    # Access the dataframe to get record_ids
    # If max_samples was used, the dataset.df is already sliced
    test_df = test_dataset.df

    current_idx = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            # Forward pass
            # Model returns sigmoid probabilities
            preds = model(images)

            # Apply threshold to get binary mask
            # Shape: (B, 1, H, W) -> (B, H, W)
            preds_binary = (preds > threshold).squeeze(1).cpu().numpy().astype(np.uint8)

            batch_len = preds_binary.shape[0]

            for i in range(batch_len):
                # Get record_id
                record_id = str(test_df.iloc[current_idx]["record_id"])

                # Encode
                mask = preds_binary[i]
                rle_str = rle_encode(mask)

                results.append({"record_id": record_id, "encoded_pixels": rle_str})

                current_idx += 1

    # 5. Save Submission
    submission_df = pd.DataFrame(results)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Total predictions: {len(submission_df)}")
