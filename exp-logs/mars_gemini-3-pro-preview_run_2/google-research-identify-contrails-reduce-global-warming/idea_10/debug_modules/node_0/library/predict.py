import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, rle_encode
from library.dataset import ContrailDataset, get_transforms
from library.model import CascadedUNet


def predict_with_tta(model, images):
    """
    Performs Test-Time Augmentation (TTA) by averaging predictions
    from Original, Horizontal Flip, Vertical Flip, and 180-degree Rotation.

    Args:
        model (nn.Module): The trained model.
        images (torch.Tensor): Batch of images (B, C, H, W).

    Returns:
        torch.Tensor: Averaged probability map (B, 1, H, W).
    """
    # 1. Original View
    # Model returns (stage1_logits, stage2_logits), we use stage2 for final output
    logits_orig = model(images)[1]
    probs_orig = torch.sigmoid(logits_orig)

    # 2. Horizontal Flip (dim 3)
    images_h = torch.flip(images, dims=[3])
    logits_h = model(images_h)[1]
    probs_h = torch.sigmoid(logits_h)
    probs_h = torch.flip(probs_h, dims=[3])  # Flip back

    # 3. Vertical Flip (dim 2)
    images_v = torch.flip(images, dims=[2])
    logits_v = model(images_v)[1]
    probs_v = torch.sigmoid(logits_v)
    probs_v = torch.flip(probs_v, dims=[2])  # Flip back

    # 4. Rotate 180 (dims 2, 3)
    # k=2 means 2 * 90 = 180 degrees
    images_r = torch.rot90(images, k=2, dims=[2, 3])
    logits_r = model(images_r)[1]
    probs_r = torch.sigmoid(logits_r)
    probs_r = torch.rot90(probs_r, k=-2, dims=[2, 3])  # Rotate back

    # Average predictions
    avg_probs = (probs_orig + probs_h + probs_v + probs_r) / 4.0

    return avg_probs


def generate_submission(debug=False):
    """
    Main inference function.
    Loads the best model, iterates over the test dataset, applies TTA (if configured),
    encodes predictions, and saves the submission CSV.

    Args:
        debug (bool): If True, processes only a small subset of the test data.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Starting inference on device: {device}")

    # 2. Data Preparation
    if not os.path.exists(Config.TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"Test metadata not found at {Config.TEST_METADATA_PATH}"
        )

    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    if debug:
        print(f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} test records.")
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    print(f"Found {len(test_df)} test records.")

    # Dataset and Loader
    # Use 'validation' transforms for test (just ToTensorV2)
    test_dataset = ContrailDataset(
        test_df, split="test", transform=get_transforms("validation")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # Crucial to maintain order matching test_df
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Loading
    print("Loading Cascaded ResNet18 U-Net...")
    model = CascadedUNet().to(device)

    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.BEST_MODEL_PATH}"
        )

    checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 4. Inference Loop
    submission_rows = []

    # We need to map predictions back to record_ids.
    # Since shuffle=False, the loader order matches test_df order.
    record_ids = test_df["record_id"].astype(str).values
    current_idx = 0

    print("Running predictions...")
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device, non_blocking=True)
            batch_size = images.size(0)

            # Predict
            if Config.USE_TTA:
                probs = predict_with_tta(model, images)
            else:
                # Standard inference
                _, logits = model(images)
                probs = torch.sigmoid(logits)

            # Move to CPU for post-processing
            probs_np = probs.cpu().numpy()  # Shape (B, 1, H, W)

            # Process batch
            for i in range(batch_size):
                # Get corresponding record_id
                rid = record_ids[current_idx]
                current_idx += 1

                # Extract mask for this image
                # Shape (1, H, W) -> (H, W)
                prob_map = probs_np[i, 0]

                # Threshold
                binary_mask = (prob_map > Config.THRESHOLD).astype(np.uint8)

                # RLE Encode
                encoded_string = rle_encode(binary_mask)

                submission_rows.append(
                    {"record_id": rid, "encoded_pixels": encoded_string}
                )

    # 5. Save Submission
    submission_df = pd.DataFrame(submission_rows)

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    output_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
