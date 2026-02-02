import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import set_seed, rle_encode
from library.dataset import ContrailDataset
from library.model import ConvNeXtUNet


def predict_with_tta(model, images, device):
    """
    Performs inference using Test-Time Augmentation (TTA).
    Augmentations: Original, Horizontal Flip, Vertical Flip, 180-degree Rotation.

    Args:
        model (nn.Module): The trained model.
        images (torch.Tensor): Batch of input images (N, C, H, W).
        device (torch.device): Computation device.

    Returns:
        torch.Tensor: Averaged probability maps (N, 1, H, W).
    """
    # Ensure images are on the correct device
    images = images.to(device, dtype=torch.float32)

    # 1. Original
    with torch.no_grad():
        logits_orig = model(images)
        probs_orig = torch.sigmoid(logits_orig)

    # 2. Horizontal Flip (dim 3 is width)
    images_h = torch.flip(images, dims=[3])
    with torch.no_grad():
        logits_h = model(images_h)
        probs_h = torch.sigmoid(logits_h)
        # Flip back
        probs_h = torch.flip(probs_h, dims=[3])

    # 3. Vertical Flip (dim 2 is height)
    images_v = torch.flip(images, dims=[2])
    with torch.no_grad():
        logits_v = model(images_v)
        probs_v = torch.sigmoid(logits_v)
        # Flip back
        probs_v = torch.flip(probs_v, dims=[2])

    # 4. 180-degree Rotation (Horizontal + Vertical Flip)
    images_rot = torch.flip(images, dims=[2, 3])
    with torch.no_grad():
        logits_rot = model(images_rot)
        probs_rot = torch.sigmoid(logits_rot)
        # Flip back
        probs_rot = torch.flip(probs_rot, dims=[2, 3])

    # Average the probabilities
    avg_probs = (probs_orig + probs_h + probs_v + probs_rot) / 4.0

    return avg_probs


def inference():
    """
    Main inference routine.
    Loads data, model, generates predictions, and saves submission.csv.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Load Metadata
    if not os.path.exists(Config.TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"Test metadata not found at {Config.TEST_METADATA_PATH}"
        )

    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # 3. Dataset & Loader
    # Note: ContrailDataset handles caching logic internally
    test_dataset = ContrailDataset(test_df, stage="test", load_cached_data=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # 4. Load Model
    model = ConvNeXtUNet()
    model.to(device)

    if os.path.exists(Config.BEST_MODEL_PATH):
        checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print(
            f"Warning: Model checkpoint not found at {Config.BEST_MODEL_PATH}. Using random weights."
        )

    model.eval()

    # 5. Prediction Loop
    results = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for images, _, record_ids in test_loader:
            # Predict with TTA
            # Output shape: (B, 1, H, W)
            probs = predict_with_tta(model, images, device)

            # Apply threshold
            preds = (probs > Config.THRESHOLD).float()

            # Move to CPU and convert to numpy for encoding
            preds_np = preds.cpu().numpy()

            # Iterate over batch
            for i in range(preds_np.shape[0]):
                # Extract single mask: (H, W)
                mask = preds_np[i, 0, :, :]

                # Run-Length Encode
                encoded_string = rle_encode(mask)

                # Store result
                results.append(
                    {"record_id": record_ids[i], "encoded_pixels": encoded_string}
                )

    # 6. Save Submission
    submission_df = pd.DataFrame(results)

    # Ensure columns are in correct order (though CSV doesn't strictly enforce column order, it's good practice)
    # The sample submission usually has record_id, encoded_pixels
    submission_df = submission_df[["record_id", "encoded_pixels"]]

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(
        f"Submission saved to {Config.SUBMISSION_PATH} with {len(submission_df)} records."
    )
