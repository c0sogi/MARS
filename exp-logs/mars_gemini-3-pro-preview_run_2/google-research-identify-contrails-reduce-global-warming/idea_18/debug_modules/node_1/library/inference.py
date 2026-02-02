import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import ContrailDataset
from library.network import AttentionGatedUNet
from library.utils import rle_encode


def predict_and_submit(
    checkpoint_path=None, batch_size=None, device=Config.DEVICE, max_samples=None
):
    """
    Performs inference on the test set using Test-Time Augmentation (TTA),
    generates RLE-encoded predictions, and saves the submission file.

    Args:
        checkpoint_path (str, optional): Path to the trained model weights.
                                         Defaults to Config.WORKING_DIR/best_model.pth.
        batch_size (int, optional): Batch size for inference. Defaults to Config.BATCH_SIZE * 2.
        device (str): Device to run inference on ('cuda' or 'cpu').
        max_samples (int, optional): Limit number of test samples for debugging.
    """
    # 1. Setup
    Config.set_seed(Config.SEED)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if batch_size is None:
        # Inference consumes less memory than training, so we can increase batch size
        batch_size = Config.BATCH_SIZE * 2

    print(f"Starting Inference...")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    # 2. Load Model
    model = AttentionGatedUNet(
        encoder_name=Config.ENCODER_NAME,
        encoder_weights=None,  # No need to download pretrained weights, we load checkpoint
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. Prepare Data
    test_dataset = ContrailDataset(split="test", max_samples=max_samples)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    results = []

    # 4. Inference Loop with TTA
    print(f"Processing {len(test_dataset)} test images...")

    with torch.no_grad():
        for images, record_ids in test_loader:
            images = images.to(device, dtype=torch.float32)

            # --- TTA View 1: Original ---
            logits_1 = model(images)
            probs_1 = torch.sigmoid(logits_1)

            # --- TTA View 2: Horizontal Flip ---
            # Flip width (dim 3)
            images_h = torch.flip(images, dims=[3])
            logits_h = model(images_h)
            # Flip back
            probs_2 = torch.flip(torch.sigmoid(logits_h), dims=[3])

            # --- TTA View 3: Vertical Flip ---
            # Flip height (dim 2)
            images_v = torch.flip(images, dims=[2])
            logits_v = model(images_v)
            # Flip back
            probs_3 = torch.flip(torch.sigmoid(logits_v), dims=[2])

            # --- TTA View 4: 180 Degree Rotation ---
            # Rotate 180 (k=2) on spatial dims (2, 3)
            images_r = torch.rot90(images, k=2, dims=[2, 3])
            logits_r = model(images_r)
            # Rotate back (k=-2)
            probs_4 = torch.rot90(torch.sigmoid(logits_r), k=-2, dims=[2, 3])

            # --- Ensemble ---
            avg_probs = (probs_1 + probs_2 + probs_3 + probs_4) / 4.0

            # Move to CPU for post-processing
            avg_probs = avg_probs.cpu().numpy()

            # 5. Encode Predictions
            # avg_probs shape is (B, 1, H, W)
            for i in range(len(record_ids)):
                rec_id = record_ids[i]
                prob_map = avg_probs[i, 0, :, :]  # (H, W)

                # Thresholding
                binary_mask = (prob_map > Config.THRESHOLD).astype(np.uint8)

                # RLE Encoding
                rle_str = rle_encode(binary_mask)

                results.append({"record_id": rec_id, "encoded_pixels": rle_str})

    # 6. Create Submission
    submission_df = pd.DataFrame(results)

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Total predictions generated: {len(submission_df)}")
