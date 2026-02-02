import os
import torch
from torch.utils.data import DataLoader

from library.config import Config, seed_everything
from library.model import MultiTaskCenterNet
from library.dataset import VinBigDataDataset
from library.engine import generate_submission


def predict_and_format(
    model_path=Config.MODEL_SAVE_PATH,
    batch_size=Config.BATCH_SIZE,
    device=Config.DEVICE,
    num_workers=Config.NUM_WORKERS,
):
    """
    Orchestrates the inference pipeline for the Thoracic Disease Detection task.

    This function:
    1. Sets random seeds for reproducibility.
    2. Initializes the test dataset and dataloader using metadata from ./metadata.
    3. Loads the MultiTaskCenterNet architecture and restores weights from the checkpoint.
    4. Calls the engine's generate_submission function to:
       - Run the model on the test set.
       - Apply the Global Classification Gate (Finding vs No Finding).
       - Decode heatmap peaks into bounding boxes.
       - Rescale boxes to the original image dimensions.
       - Save the formatted predictions to ./submission/submission.csv.

    Args:
        model_path (str): Path to the saved model checkpoint (.pth).
        batch_size (int): Batch size for the data loader.
        device (str): Device to run inference on ('cuda' or 'cpu').
        num_workers (int): Number of worker processes for data loading.
    """
    # 1. Ensure Determinism
    seed_everything(Config.SEED)

    print(f"Starting inference pipeline on device: {device}")

    # 2. Prepare Data
    # The dataset class handles reading ./metadata/test_meta.csv and
    # uses the centralized dicom_utils.load_image_and_metadata for robust parsing.
    print("Initializing test dataset...")
    test_dataset = VinBigDataDataset(split="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
        drop_last=False,
    )
    print(f"Test dataset loaded with {len(test_dataset)} samples.")

    # 3. Prepare Model
    print("Initializing MultiTaskCenterNet model...")
    model = MultiTaskCenterNet()
    model.to(device)

    # 4. Load Weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    print(f"Loading model weights from {model_path}...")
    # Load checkpoint with map_location to handle CPU/GPU transparency
    checkpoint = torch.load(model_path, map_location=device)

    # Handle cases where checkpoint is a dict containing 'state_dict' or just the state_dict itself
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # 5. Execute Prediction and Formatting
    # The generate_submission function in library.engine encapsulates the logic for:
    # - Iterating over the loader
    # - Checking the Global Classification Head (Finding vs No Finding)
    # - Decoding boxes if Findings are present
    # - Rescaling boxes from 512x512 to Original Dimensions
    # - Writing the final submission.csv
    generate_submission(model, test_loader, device)

    print("Inference pipeline completed successfully.")
