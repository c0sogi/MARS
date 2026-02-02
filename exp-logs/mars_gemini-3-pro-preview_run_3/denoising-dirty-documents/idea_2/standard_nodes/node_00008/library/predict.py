import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config, seed_everything
from library.model import UNet
from library.dataset import load_data, DenoisingDataset
from library.utils import load_checkpoint, generate_submission_file


def generate_predictions(
    model_path=Config.MODEL_SAVE_PATH,
    test_metadata_path=Config.TEST_METADATA_PATH,
    batch_size=1,
    device=Config.DEVICE,
    load_cached_data=True,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Loads the model and performs inference on the test dataset.

    Args:
        model_path (str): Path to the trained model checkpoint.
        test_metadata_path (str): Path to the test metadata CSV.
        batch_size (int): Batch size for inference (default 1 for variable image sizes).
        device (torch.device): Device to run inference on.
        load_cached_data (bool): Whether to use cached data if available.
        debug_sample_size (int, optional): Limit dataset size for debugging.

    Returns:
        dict: A dictionary mapping image_id to predicted numpy arrays.
    """
    seed_everything(Config.SEED)

    print(f"Loading test data from {test_metadata_path}...")
    # Load test data directly using the library function which handles caching
    test_ids, test_inputs = load_data(
        test_metadata_path, "input_path", "test_in", load_cached_data, debug_sample_size
    )

    # Create Dataset and Loader
    # train_mode=False ensures we get full images without random cropping
    test_dataset = DenoisingDataset(test_inputs, train_mode=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Initialize Model
    print("Initializing model...")
    model = UNet(n_channels=Config.IN_CHANNELS, n_classes=Config.OUT_CHANNELS).to(
        device
    )

    # Load Weights
    print(f"Loading checkpoint from {model_path}...")
    checkpoint = load_checkpoint(model_path, model, device=device)
    if checkpoint is None:
        print(
            f"Warning: No checkpoint found at {model_path}. Using random initialization."
        )
    else:
        print(
            f"Model loaded. Epoch: {checkpoint['epoch']}, Validation Loss: {checkpoint['loss']}"
        )

    model.eval()
    predictions = {}

    print(f"Starting inference on {len(test_dataset)} images...")

    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device)

            # U-Net requires input dimensions to be divisible by 16 due to max-pooling layers.
            # We calculate necessary padding.
            h, w = inputs.shape[2], inputs.shape[3]
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16

            if pad_h > 0 or pad_w > 0:
                inputs_padded = F.pad(inputs, (0, pad_w, 0, pad_h), mode="reflect")
            else:
                inputs_padded = inputs

            # Forward pass
            outputs_padded = model(inputs_padded)

            # Crop back to original dimensions
            outputs = outputs_padded[:, :, :h, :w]

            # Clamp values to valid range [0, 1]
            outputs = torch.clamp(outputs, 0, 1)

            # Convert to numpy array (H, W)
            # Squeeze removes batch and channel dimensions (1, 1, H, W) -> (H, W)
            pred_img = outputs.squeeze().cpu().numpy()

            # Store prediction
            img_id = test_ids[i]
            predictions[img_id] = pred_img

    return predictions


def format_submission(predictions, output_path=Config.SUBMISSION_PATH):
    """
    Formats the predictions dictionary into the submission CSV format.

    Args:
        predictions (dict): Dictionary of {image_id: numpy_array}.
        output_path (str): Path to save the CSV.
    """
    print(f"Formatting and saving submission to {output_path}...")
    generate_submission_file(predictions, output_path)
    print("Submission saved successfully.")


def run_inference(
    model_path=Config.MODEL_SAVE_PATH,
    output_path=Config.SUBMISSION_PATH,
    load_cached_data=True,
):
    """
    Main entry point to run the full inference pipeline.
    """
    # Generate predictions
    predictions = generate_predictions(
        model_path=model_path, load_cached_data=load_cached_data
    )

    # Save submission
    format_submission(predictions, output_path)
