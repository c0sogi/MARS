import os
import torch
import numpy as np
import pandas as pd
import cv2
from library.config import Config
from library.model import ResDnCNN
from library.utils import save_submission
from library.data_loader import load_image


def augment_tensor(img_tensor, k, flip):
    """
    Applies geometric transformations to a tensor.
    Args:
        img_tensor (torch.Tensor): Input tensor [B, C, H, W]
        k (int): Number of 90-degree rotations (0, 1, 2, 3)
        flip (bool): Whether to apply horizontal flip
    Returns:
        torch.Tensor: Transformed tensor
    """
    # Apply flip first if requested
    if flip:
        # Flip along width axis (axis 3)
        img_tensor = torch.flip(img_tensor, [3])

    # Apply rotation
    if k > 0:
        # Rotate in the plane defined by H and W (axes 2 and 3)
        img_tensor = torch.rot90(img_tensor, k, [2, 3])

    return img_tensor


def inverse_augment_tensor(img_tensor, k, flip):
    """
    Reverses the geometric transformations applied by augment_tensor.
    Note: Operations must be reversed in reverse order.
    Inverse of (Rot(Flip(x))) is Flip(Rot_inv(y)).
    """
    # Reverse rotation first
    if k > 0:
        # Rotate by -k (or 4-k)
        img_tensor = torch.rot90(img_tensor, -k, [2, 3])

    # Reverse flip
    if flip:
        img_tensor = torch.flip(img_tensor, [3])

    return img_tensor


def geometric_self_ensemble(model, x, device):
    """
    Performs inference using Geometric Self-Ensemble (8 transforms).
    Args:
        model (nn.Module): The trained model
        x (torch.Tensor): Input image tensor [1, 1, H, W]
        device (str): Device to run inference on
    Returns:
        torch.Tensor: Averaged noise prediction [1, 1, H, W]
    """
    noise_accum = torch.zeros_like(x)
    count = 0

    # Iterate over all 8 combinations of flips and rotations
    # Flips: [False, True]
    # Rotations: [0, 1, 2, 3]
    for flip in [False, True]:
        for k in range(4):
            # 1. Transform input
            x_aug = augment_tensor(x, k, flip)
            x_aug = x_aug.to(device)

            # 2. Predict noise
            with torch.no_grad():
                noise_aug = model(x_aug)

            # 3. Inverse transform prediction
            noise_pred = inverse_augment_tensor(noise_aug, k, flip)

            # 4. Accumulate (keep on CPU to save GPU memory if needed, though tensors are small here)
            noise_accum += noise_pred.cpu()
            count += 1

    # Average the predictions
    return noise_accum / count


def predict_and_save():
    """
    Main inference function.
    Loads data, loads model, generates predictions via TTA, and saves submission.
    """
    print("Starting Inference Process...")

    # 1. Setup
    device = torch.device(Config.DEVICE)
    print(f"Inference Device: {device}")

    # 2. Load Metadata
    test_df = pd.read_csv(Config.TEST_CSV)
    print(f"Found {len(test_df)} test images.")

    # 3. Load Model
    print("Loading model...")
    model = ResDnCNN().to(device)

    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}. Train the model first."
        )

    state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Inference Loop
    predictions = {}

    print("Processing images...")
    for idx, row in test_df.iterrows():
        image_id = row["image_id"]
        input_path = row["input_path"]

        # Load image (returns numpy array [H, W] in range [0, 1])
        # load_image handles full path construction via Config.INPUT_DIR
        img_in = load_image(input_path)

        # Prepare tensor: [H, W] -> [1, 1, H, W]
        img_tensor = torch.from_numpy(img_in).float().unsqueeze(0).unsqueeze(0)

        # Predict noise using Geometric Self-Ensemble
        if Config.TTA_ENABLED:
            predicted_noise_tensor = geometric_self_ensemble(model, img_tensor, device)
        else:
            # Single pass fallback
            img_tensor = img_tensor.to(device)
            with torch.no_grad():
                predicted_noise_tensor = model(img_tensor)
            predicted_noise_tensor = predicted_noise_tensor.cpu()

        # Convert back to numpy
        predicted_noise = predicted_noise_tensor.squeeze().numpy()

        # Reconstruct clean image: Clean = Noisy - Noise
        pred_clean = img_in - predicted_noise

        # Store result
        predictions[image_id] = pred_clean

        if (idx + 1) % 5 == 0:
            print(f"Processed {idx + 1}/{len(test_df)} images.")

    # 5. Save Submission
    print("Saving submission...")
    save_submission(predictions, Config.SUBMISSION_PATH)
    print("Inference complete.")
