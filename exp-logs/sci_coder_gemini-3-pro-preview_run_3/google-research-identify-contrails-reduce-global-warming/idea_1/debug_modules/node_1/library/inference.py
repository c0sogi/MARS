import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.data import get_dataloaders
from library.model import LightUNet
from library.utils import rle_encode, set_seed


def generate_submission(
    model_path=os.path.join(Config.WORKING_DIR, "best_model.pth"),
    batch_size=Config.BATCH_SIZE,
    debug=Config.DEBUG,
    threshold=Config.THRESHOLD,
):
    """
    Loads the trained model and generates predictions for the test set.
    Saves the result to the submission file in RLE format.

    Args:
        model_path (str): Path to the trained model checkpoint (.pth file).
        batch_size (int): Batch size for inference.
        debug (bool): If True, runs on a subset of the test data.
        threshold (float): Probability threshold for converting model output to binary mask.

    Returns:
        pd.DataFrame: The submission dataframe containing record_ids and encoded pixels.
    """
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    device = torch.device(Config.DEVICE)

    # Get Test DataLoader
    # get_dataloaders returns (train_loader, val_loader, test_loader)
    # We only need the test_loader
    _, _, test_loader = get_dataloaders(
        batch_size=batch_size, num_workers=Config.NUM_WORKERS, debug=debug
    )

    # Initialize Model
    model = LightUNet().to(device)

    # Load Model Weights
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Successfully loaded model weights from {model_path}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return pd.DataFrame()  # Return empty DF on failure
    else:
        print(
            f"Warning: Model weights not found at {model_path}. Using random initialization (expect poor performance)."
        )

    model.eval()

    submission_data = []

    print("Starting inference on test set...")

    with torch.no_grad():
        for images, _, record_ids in test_loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Apply threshold to get binary mask
            # outputs shape: (Batch, 1, H, W)
            preds = (outputs > threshold).float().cpu().numpy()

            # Encode each image in the batch
            for i, record_id in enumerate(record_ids):
                # Extract single mask: (1, H, W) -> (H, W)
                mask = preds[i, 0, :, :]

                # Run-Length Encoding
                rle = rle_encode(mask)

                submission_data.append({"record_id": record_id, "encoded_pixels": rle})

    # Create DataFrame
    df = pd.DataFrame(submission_data)

    # Save Submission
    try:
        df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    except Exception as e:
        print(f"Error saving submission file: {e}")

    return df
