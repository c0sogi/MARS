import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from library import config
from library import utils
from library import dataset
from library import model as lib_model


def predict_tta(model, images, device):
    """
    Performs Test-Time Augmentation (TTA) inference.

    Strategy:
    1. Forward pass on the original image tensor.
    2. Forward pass on the horizontally flipped tensor.
    3. Forward pass on the vertically flipped tensor.
    4. Average the sigmoid probabilities from all three passes.

    Args:
        model (nn.Module): The trained PyTorch model.
        images (torch.Tensor): Input batch of images [B, C, H, W].
        device (str): Computation device.

    Returns:
        torch.Tensor: Averaged probabilities [B, 1].
    """
    model.eval()
    with torch.no_grad():
        # 1. Original
        logits_orig = model(images)
        probs_orig = torch.sigmoid(logits_orig)

        # 2. Horizontal Flip (dim 3 is Width in NCHW)
        images_hflip = torch.flip(images, [3])
        logits_hflip = model(images_hflip)
        probs_hflip = torch.sigmoid(logits_hflip)

        # 3. Vertical Flip (dim 2 is Height in NCHW)
        images_vflip = torch.flip(images, [2])
        logits_vflip = model(images_vflip)
        probs_vflip = torch.sigmoid(logits_vflip)

        # Average probabilities
        avg_probs = (probs_orig + probs_hflip + probs_vflip) / 3.0

    return avg_probs


def generate_submission(
    model_weights_path=None,
    output_file="submission.csv",
    batch_size=config.BATCH_SIZE,
    device=config.DEVICE,
    debug=False,
):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model_weights_path (str): Path to the trained model weights.
                                  Defaults to config.WORKING_DIR/best_model.pth.
        output_file (str): Name of the output CSV file.
        batch_size (int): Batch size for inference.
        device (str): Device to run inference on.
        debug (bool): If True, runs on a subset of data for debugging.
    """
    # 1. Setup Paths and Logging
    if model_weights_path is None:
        model_weights_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    output_path = os.path.join(submission_dir, output_file)

    logger = utils.get_logger("inference")
    logger.info(f"Starting inference pipeline.")
    logger.info(f"Model weights: {model_weights_path}")
    logger.info(f"Output path: {output_path}")

    # 2. Load Model Architecture and Weights
    logger.info("Initializing AsymmetricEfficientNet...")
    net = lib_model.AsymmetricEfficientNet()

    if os.path.exists(model_weights_path):
        utils.load_checkpoint(model_weights_path, net, device=device)
    else:
        logger.warning(
            f"Weights not found at {model_weights_path}. Using random initialization (DEBUG ONLY)."
        )

    net = net.to(device)
    net.eval()

    # 3. Prepare Test Data
    # The RSNADataset with split='test' reads from metadata/test.csv
    # It handles ROI caching internally via dicom_processing.get_roi_anchor
    logger.info("Preparing test dataset...")
    test_dataset = dataset.RSNADataset(
        split="test", transform=dataset.get_transforms("test"), debug=debug
    )

    # Shuffle must be False to ensure predictions align with the dataframe order
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    logger.info(f"Test dataset contains {len(test_dataset)} samples.")

    # 4. Run Inference
    all_probs = []

    logger.info("Running inference with TTA (Original + HFlip + VFlip)...")
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(device, dtype=torch.float32)

            # Predict
            probs = predict_tta(net, images, device)

            # Store results
            probs_np = probs.cpu().numpy().flatten()
            all_probs.extend(probs_np)

            if (i + 1) % 5 == 0:
                logger.info(f"Processed batch {i + 1}/{len(test_loader)}")

    # 5. Create Submission DataFrame
    # Retrieve the original dataframe to ensure IDs match exactly
    df_test = test_dataset.df.copy()

    # Validation check
    if len(all_probs) != len(df_test):
        logger.error(
            f"Prediction count mismatch! Got {len(all_probs)}, expected {len(df_test)}."
        )

    # Assign predictions
    df_test["MGMT_value"] = all_probs

    # Format the submission dataframe
    # We keep BraTS21ID as is (integer) based on sample_submission.csv format
    submission_df = df_test[["BraTS21ID", "MGMT_value"]]

    # 6. Save to CSV
    submission_df.to_csv(output_path, index=False)
    logger.info(f"Submission successfully saved to {output_path}")

    # Print preview
    print("\nSubmission Preview:")
    print(submission_df.head())
