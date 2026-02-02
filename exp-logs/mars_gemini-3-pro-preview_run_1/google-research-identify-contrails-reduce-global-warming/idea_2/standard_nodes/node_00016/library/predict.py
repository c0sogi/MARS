import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.model import ResNet34UNet
from library.dataset import ContrailDataset
from library.utils import set_seed, rle_encode


def generate_submission(
    checkpoint_path=None, batch_size=Config.BATCH_SIZE, device=Config.DEVICE
):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        checkpoint_path (str, optional): Path to the model checkpoint.
                                         Defaults to Config.CHECKPOINT_DIR/best_model.pth.
        batch_size (int): Batch size for inference.
        device (str): Device to run inference on ('cuda' or 'cpu').
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(device)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Loading model from {checkpoint_path}...")
    print(f"Inference device: {device}")

    # 2. Model Initialization
    # Initialize with pretrained=False because we are loading specific weights
    # and want to avoid potential connection attempts to download ImageNet weights.
    model = ResNet34UNet(
        in_channels=Config.IN_CHANNELS, out_channels=Config.CLASSES, pretrained=False
    )

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    model.to(device)
    model.eval()

    # 3. Data Loading
    # shuffle=False is critical to match predictions with record_ids via index
    test_dataset = ContrailDataset(split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Test samples: {len(test_dataset)}")

    # 4. Inference Loop
    results = []
    current_idx = 0

    print("Starting inference...")

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            # Forward pass
            logits = model(images)

            # Apply Sigmoid and Threshold
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float().cpu().numpy()  # Shape: (B, 1, H, W)

            batch_len = preds.shape[0]

            for i in range(batch_len):
                # Retrieve record_id from the dataset's dataframe using the global index
                # This works because shuffle=False
                record_id = str(test_dataset.df.iloc[current_idx + i]["record_id"])

                # Extract mask for the current image (H, W)
                mask = preds[i, 0, :, :]

                # Encode to RLE
                encoded_pixels = rle_encode(mask)

                results.append(
                    {"record_id": record_id, "encoded_pixels": encoded_pixels}
                )

            current_idx += batch_len

    # 5. Save Submission
    submission_df = pd.DataFrame(results)

    # Ensure columns are in the correct order (though not strictly enforced by CSV, good for sanity)
    submission_df = submission_df[["record_id", "encoded_pixels"]]

    save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(save_path, index=False)

    print(f"Submission saved to {save_path}")
    print(f"Total records processed: {len(submission_df)}")

    return submission_df
