import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, save_submission_file
from library.model import UNet
from library.dataset import load_processed_data, DenoisingDataset


def generate_submission(load_cached_data=True, batch_size=1):
    """
    Generates predictions for the test dataset using the trained FlatCNN model
    and saves the results to a submission file.

    Args:
        load_cached_data (bool): Whether to attempt loading pre-processed data from cache.
                                 Defaults to True.
        batch_size (int): Batch size for inference. Defaults to 1.
                          Note: Since test images may have varying dimensions, keeping
                          batch_size=1 is recommended unless a custom collate function is used.
    """
    # Ensure reproducibility
    set_seed()

    # Setup device
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # --- 1. Load Test Data ---
    # Uses the shared data loading logic which handles caching and metadata reading
    print("Loading test data...")
    test_data = load_processed_data(Config.TEST_METADATA_PATH, "test", load_cached_data)

    # Create dataset and loader
    # Mode 'test' ensures the dataset returns (noisy_image, image_id)
    test_dataset = DenoisingDataset(test_data, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # --- 2. Load Model ---
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(
            f"Error: Model file not found at {Config.MODEL_SAVE_PATH}. Cannot generate submission."
        )
        return

    print(f"Loading model from {Config.MODEL_SAVE_PATH}...")
    model = UNet().to(device)
    state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # --- 3. Run Inference ---
    predictions = {}
    print("Generating predictions...")

    with torch.no_grad():
        for batch in test_loader:
            # Unpack batch. DenoisingDataset in test mode returns (noisy_tensor, id)
            noisy, img_ids = batch

            noisy = noisy.to(device)

            # Forward pass
            # Output shape: (Batch, 1, Height, Width)
            outputs = model(noisy)

            # Process each image in the batch
            for i, img_id in enumerate(img_ids):
                # Select the i-th image in batch and remove channel dimension
                # outputs[i] is (1, H, W) -> squeeze(0) -> (H, W)
                pred_tensor = outputs[i].squeeze(0)

                # Convert to numpy array
                pred_img = pred_tensor.cpu().numpy()

                # Clip values to ensure they are within valid range [0, 1]
                pred_img = np.clip(pred_img, 0, 1)

                # Store prediction
                predictions[str(img_id)] = pred_img

    # --- 4. Save Submission ---
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    save_submission_file(predictions, Config.SUBMISSION_PATH)
    print("Submission generation complete.")
