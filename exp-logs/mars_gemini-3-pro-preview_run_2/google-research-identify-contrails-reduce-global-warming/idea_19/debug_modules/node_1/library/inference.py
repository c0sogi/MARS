import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import ContrailDataset, get_transforms
from library.model import AttentionGatedConvNeXtUNet
from library.utils import rle_encode


def predict_with_tta(model, images):
    """
    Performs Test-Time Augmentation (TTA) prediction.
    Augmentations: Original, Horizontal Flip, Vertical Flip, Rotate 180.

    Args:
        model (nn.Module): The trained model.
        images (torch.Tensor): Input batch of images (B, C, H, W).

    Returns:
        torch.Tensor: Averaged probability maps.
    """
    model.eval()

    # Sigmoid for probability conversion
    sigmoid = torch.nn.Sigmoid()

    with torch.no_grad():
        # 1. Original
        logits_orig = model(images)
        probs_orig = sigmoid(logits_orig)

        # 2. Horizontal Flip
        # Flip along width (dim 3)
        images_h = torch.flip(images, dims=[3])
        logits_h = model(images_h)
        probs_h = sigmoid(logits_h)
        probs_h = torch.flip(probs_h, dims=[3])  # Flip back

        # 3. Vertical Flip
        # Flip along height (dim 2)
        images_v = torch.flip(images, dims=[2])
        logits_v = model(images_v)
        probs_v = sigmoid(logits_v)
        probs_v = torch.flip(probs_v, dims=[2])  # Flip back

        # 4. Rotate 180 (equivalent to H-flip + V-flip)
        images_rot = torch.flip(images, dims=[2, 3])
        logits_rot = model(images_rot)
        probs_rot = sigmoid(logits_rot)
        probs_rot = torch.flip(probs_rot, dims=[2, 3])  # Rotate back

        # Average probabilities
        avg_probs = (probs_orig + probs_h + probs_v + probs_rot) / 4.0

    return avg_probs


def run_inference(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    device=Config.DEVICE,
    threshold=Config.THRESHOLD,
):
    """
    Main inference routine.
    Loads the best model, iterates over the test set, performs TTA,
    encodes predictions, and saves the submission file.

    Args:
        batch_size (int): Batch size for inference.
        num_workers (int): Number of dataloader workers.
        device (str): Computation device ('cuda' or 'cpu').
        threshold (float): Threshold for binarizing probabilities.
    """
    print("Starting Inference...")

    # 1. Setup Data
    # Transforms for test are just ToTensorV2 (handled in dataset.py via get_transforms('test'))
    test_transforms = get_transforms("test")
    test_dataset = ContrailDataset(split="test", transform=test_transforms)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 2. Setup Model
    model = AttentionGatedConvNeXtUNet()
    model.to(device)

    # Load weights
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found at {Config.BEST_MODEL_PATH}")

    checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Loaded model weights from {Config.BEST_MODEL_PATH}")

    # 3. Inference Loop
    submission_data = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            record_ids = batch["record_id"]

            # Predict with TTA if enabled in Config, else single pass
            if Config.USE_TTA:
                probs = predict_with_tta(model, images)
            else:
                model.eval()
                probs = torch.sigmoid(model(images))

            # Binarize
            preds = (probs > threshold).float()

            # Move to CPU for encoding
            preds_np = preds.cpu().numpy()  # Shape: (B, 1, H, W)

            # Encode each image in the batch
            for b in range(preds_np.shape[0]):
                # Extract (H, W) mask
                mask = preds_np[b, 0, :, :]
                rle = rle_encode(mask)

                submission_data.append(
                    {"record_id": record_ids[b], "encoded_pixels": rle}
                )

    # 4. Save Submission
    df_submission = pd.DataFrame(submission_data)

    # Ensure directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Total records processed: {len(df_submission)}")
