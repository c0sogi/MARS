import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import get_device
from library.model import AsymmetricEfficientNet
from library.data_loader import get_dataloaders


def predict_tta(model, images, device):
    """
    Performs Test-Time Augmentation (TTA) inference.

    Strategies:
    1. Original Input
    2. Horizontal Flip
    3. Vertical Flip

    The final probability is the arithmetic mean of these three passes.

    Args:
        model (nn.Module): The trained model in eval mode.
        images (torch.Tensor): Batch of images (B, C, H, W).
        device (torch.device): Compute device.

    Returns:
        np.ndarray: Averaged probabilities for the batch.
    """
    # 1. Original
    logits_orig = model(images)
    probs_orig = torch.sigmoid(logits_orig)

    # 2. Horizontal Flip (dim 3 is width)
    images_hflip = torch.flip(images, dims=[3])
    logits_hflip = model(images_hflip)
    probs_hflip = torch.sigmoid(logits_hflip)

    # 3. Vertical Flip (dim 2 is height)
    images_vflip = torch.flip(images, dims=[2])
    logits_vflip = model(images_vflip)
    probs_vflip = torch.sigmoid(logits_vflip)

    # Average
    avg_probs = (probs_orig + probs_hflip + probs_vflip) / 3.0

    return avg_probs.cpu().numpy().flatten()


def generate_submission(load_cached_data=True):
    """
    Generates the submission file for the test set.

    Steps:
    1. Loads the best model checkpoint.
    2. Initializes the test DataLoader (guaranteed sequential order).
    3. Iterates through the test set using TTA.
    4. Matches predictions with BraTS21IDs from metadata.
    5. Saves to submission.csv.

    Args:
        load_cached_data (bool): Whether to use cached ROI anchors.
    """
    device = get_device()
    print(f"Inference Device: {device}")

    # 1. Load Data
    # We only need the test loader. get_dataloaders handles ROI caching.
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data)

    # Load test metadata to retrieve IDs (DataLoader is sequential, shuffle=False)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Verify alignment
    if len(test_loader.dataset) != len(test_df):
        print(
            f"Warning: DataLoader size ({len(test_loader.dataset)}) does not match Metadata size ({len(test_df)})."
        )

    # 2. Load Model
    print("Loading model...")
    model = AsymmetricEfficientNet().to(device)

    checkpoint_path = Config.MODEL_CHECKPOINT_PATH
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. Train the model first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different saving formats (state_dict vs full checkpoint dict)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 3. Inference Loop
    print("Starting inference with TTA...")
    all_preds = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            # Predict with TTA
            batch_preds = predict_tta(model, images, device)
            all_preds.extend(batch_preds)

    # 4. Create Submission DataFrame
    # Ensure we have exactly one prediction per row in test_df
    if len(all_preds) != len(test_df):
        print(
            f"Error: Number of predictions ({len(all_preds)}) does not match number of test samples ({len(test_df)})."
        )
        # Truncate or pad if absolutely necessary to avoid crash, though this indicates a logic bug
        if len(all_preds) > len(test_df):
            all_preds = all_preds[: len(test_df)]
        else:
            all_preds.extend([0.5] * (len(test_df) - len(all_preds)))

    submission_df = pd.DataFrame(
        {"BraTS21ID": test_df["BraTS21ID"], "MGMT_value": all_preds}
    )

    # 5. Save
    output_path = Config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
    print(submission_df.head())
