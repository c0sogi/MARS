import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.model import DepthLinkNet
from library.dataset import SaltDataset
from library.utils import set_seed, rle_encode


def generate_submission(config=None):
    """
    Generates the submission file for the competition.
    Loads the best model, runs inference on the test set, applies post-processing,
    and saves the RLE encoded masks to a CSV file.

    Args:
        config (Config, optional): Configuration object. Defaults to None.
    """
    # Initialize Config
    if config is None:
        config = Config()

    # Set Seed
    set_seed(config.SEED)

    # Device
    device = torch.device(config.DEVICE)

    # Initialize Model
    model = DepthLinkNet(
        in_channels=config.CHANNELS, num_classes=config.NUM_CLASSES
    ).to(device)

    # Load Checkpoint
    if os.path.exists(config.CHECKPOINT_PATH):
        # Load weights
        state_dict = torch.load(config.CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        # If no checkpoint exists, the model uses random initialization (mostly for debugging flow)
        pass

    model.eval()

    # Initialize Test Dataset and Loader
    # load_cached_data=True allows using cached numpy arrays if available
    test_ds = SaltDataset(
        metadata_path=config.TEST_CSV, config=config, mode="test", load_cached_data=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE * 2,  # Can use larger batch size for inference
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    results = []

    # Calculate cropping indices to revert padding
    # Model input: 128x128 -> Output: 101x101
    h_orig, w_orig = config.ORIG_SHAPE
    h_in, w_in = config.INPUT_SHAPE

    pad_top = (h_in - h_orig) // 2
    pad_left = (w_in - w_orig) // 2

    # Inference Loop with TTA (Cite solution_lesson_node_00002)
    with torch.no_grad():
        for images, depths, ids in test_loader:
            images = images.to(device)
            depths = depths.to(device)

            # Forward pass
            outputs = model(images, depths)

            # TTA: Flip
            images_flip = torch.flip(images, [3])
            outputs_flip = model(images_flip, depths)
            outputs_flip = torch.flip(outputs_flip, [3])

            # Average
            outputs_avg = (outputs + outputs_flip) / 2.0

            # Sigmoid to get probabilities
            probs = torch.sigmoid(outputs_avg)

            # Move to CPU for post-processing
            probs = probs.cpu().numpy()

            # Iterate through batch
            for i, img_id in enumerate(ids):
                # Extract single probability map (C, H, W) -> (H, W)
                # Shape is (1, 128, 128) -> (128, 128)
                prob_map = probs[i, 0, :, :]

                # Center Crop to original size (101, 101)
                mask_map = prob_map[
                    pad_top : pad_top + h_orig, pad_left : pad_left + w_orig
                ]

                # Apply Threshold
                binary_mask = (mask_map > config.THRESHOLD).astype(np.uint8)

                # RLE Encode
                rle = rle_encode(binary_mask)

                results.append({"id": img_id, "rle_mask": rle})

    # Create DataFrame
    sub_df = pd.DataFrame(results)

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

    # Save to CSV
    sub_df.to_csv(config.SUBMISSION_PATH, index=False)
