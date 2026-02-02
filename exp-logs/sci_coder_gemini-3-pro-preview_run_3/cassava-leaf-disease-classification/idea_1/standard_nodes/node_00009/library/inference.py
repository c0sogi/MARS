import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import get_dataset
from library.model import CassavaModel


def run_inference(
    checkpoint_path: str = Config.MODEL_CHECKPOINT_PATH,
    output_path: str = Config.SUBMISSION_PATH,
    debug: bool = Config.DEBUG,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
):
    """
    Runs inference on the test dataset using the trained model and generates a submission file.

    Args:
        checkpoint_path (str): Path to the saved model checkpoint.
        output_path (str): Path where the submission CSV will be saved.
        debug (bool): If True, runs inference on a small subset of the test data.
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of worker processes for data loading.
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Setup device
    device = get_device()
    print(f"Inference using device: {device}")

    # --- Data Loading ---
    print("Loading test dataset...")
    # get_dataset handles the metadata loading. For 'test' split, it uses TEST_METADATA_PATH.
    test_dataset = get_dataset("test", debug=debug)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Important: Must not shuffle to match image_ids
        num_workers=num_workers,
        pin_memory=True,
    )

    # --- Model Setup ---
    print(f"Initializing model and loading weights from {checkpoint_path}...")
    # Initialize the model architecture
    model = CassavaModel(pretrained=Config.PRETRAINED, num_classes=Config.NUM_CLASSES)

    # Load the state dictionary
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    # --- Prediction Loop ---
    print("Starting prediction loop...")
    all_preds = []

    with torch.no_grad():
        for images in test_loader:
            # Move images to device
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Get predicted class (argmax)
            # outputs shape: (batch_size, num_classes)
            _, predicted = torch.max(outputs, 1)

            # Store predictions
            all_preds.append(predicted.cpu().numpy())

    # Concatenate all batch predictions
    if len(all_preds) > 0:
        final_predictions = np.concatenate(all_preds)
    else:
        final_predictions = np.array([])

    # --- Generate Submission ---
    print("Generating submission file...")

    # Retrieve image_ids from the dataset's dataframe
    # The dataset preserves the order of the metadata file
    image_ids = test_dataset.df["image_id"].values

    # Verify lengths match
    if len(image_ids) != len(final_predictions):
        raise ValueError(
            f"Mismatch between number of images ({len(image_ids)}) and predictions ({len(final_predictions)})"
        )

    # Create submission DataFrame
    submission_df = pd.DataFrame({"image_id": image_ids, "label": final_predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print("Inference complete.")
