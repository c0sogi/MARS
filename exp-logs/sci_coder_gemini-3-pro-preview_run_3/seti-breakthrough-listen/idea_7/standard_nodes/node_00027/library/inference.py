import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from library.config import Config
from library.dataset import get_test_loader
from library.utils import seed_everything


def predict_tta(model, stream_a, stream_b, device):
    """
    Performs Test Time Augmentation (TTA) on a batch of inputs.

    Generates predictions for:
    1. Original
    2. Horizontal Flip (Time dimension)
    3. Vertical Flip (Frequency dimension)
    4. Horizontal + Vertical Flip

    Crucially, this function handles padding correctly by unpadding the input
    to the original signal size before flipping, and re-padding afterwards.
    This ensures zero-padding remains at the bottom of the spectrogram,
    matching the training distribution.

    Args:
        model (nn.Module): The trained model.
        stream_a (torch.Tensor): On-Target input batch (B, C, H, W).
        stream_b (torch.Tensor): Off-Target input batch (B, C, H, W).
        device (str): Device to perform inference on.

    Returns:
        np.ndarray: Averaged probabilities for the batch (B,).
    """
    # Dimensions
    orig_h = Config.ORIGINAL_HEIGHT
    target_h = Config.IMG_HEIGHT
    pad_amount = target_h - orig_h

    # Move inputs to device
    stream_a = stream_a.to(device)
    stream_b = stream_b.to(device)

    # 1. Unpad: Extract the valid signal region (0 to 273)
    # The dataset pads at the end of the H dimension (index 2)
    raw_a = stream_a[:, :, :orig_h, :]
    raw_b = stream_b[:, :, :orig_h, :]

    # Define TTA Transformations
    # dim 2 = Height (Frequency), dim 3 = Width (Time)
    transforms = [
        lambda x: x,  # Original
        lambda x: torch.flip(x, dims=[3]),  # Horizontal Flip
        lambda x: torch.flip(x, dims=[2]),  # Vertical Flip
        lambda x: torch.flip(x, dims=[2, 3]),  # HV Flip
    ]

    probs_list = []

    with torch.no_grad():
        for t in transforms:
            # Apply transformation to the raw signal
            aug_a = t(raw_a)
            aug_b = t(raw_b)

            # 2. Re-pad: Add padding back to the bottom to reach target height (288)
            # F.pad tuple format: (left, right, top, bottom)
            if pad_amount > 0:
                aug_a = F.pad(aug_a, (0, 0, 0, pad_amount), mode="constant", value=0)
                aug_b = F.pad(aug_b, (0, 0, 0, pad_amount), mode="constant", value=0)

            # Inference
            logits = model(aug_a, aug_b)
            probs = torch.sigmoid(logits)
            probs_list.append(probs)

    # Stack predictions and calculate mean
    # stacked shape: (4, B, 1)
    stacked_probs = torch.stack(probs_list)
    avg_probs = torch.mean(stacked_probs, dim=0)

    return avg_probs.cpu().numpy().flatten()


def generate_submission(model, device, output_path="./submission/submission.csv"):
    """
    Runs the inference pipeline on the test set and generates a submission file.

    Args:
        model (nn.Module): The trained model.
        device (str): Device to run inference on.
        output_path (str): Path to save the submission CSV.
    """
    seed_everything(Config.SEED)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load Test Data
    loader = get_test_loader()

    model.eval()
    all_preds = []

    print(f"Starting TTA Inference on {device}...")

    # Iterate over test set
    # The loader returns (data_dict, target_placeholder)
    for i, (data, _) in enumerate(loader):
        stream_a = data["stream_a"]
        stream_b = data["stream_b"]

        # Get TTA predictions for this batch
        batch_preds = predict_tta(model, stream_a, stream_b, device)
        all_preds.append(batch_preds)

    # Concatenate all batch predictions
    all_preds = np.concatenate(all_preds)

    # Load Test Metadata to map predictions to IDs
    df_test = pd.read_csv(Config.TEST_CSV)

    # Verify alignment
    if len(all_preds) != len(df_test):
        print(
            f"Warning: Prediction count ({len(all_preds)}) does not match metadata count ({len(df_test)})."
        )

    # Assign predictions
    df_test["target"] = all_preds

    # Save to CSV
    submission_df = df_test[["id", "target"]]
    submission_df.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
