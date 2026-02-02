import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.model import ShallowCNN
from library.dataset import get_dataloaders


def generate_submission(
    model_path, output_path=Config.SUBMISSION_PATH, debug=Config.DEBUG
):
    """
    Loads a trained model and generates a submission file for the test set.

    Args:
        model_path (str): Path to the .pth file containing trained model weights.
        output_path (str): Destination path for the submission CSV file.
                           Defaults to ./submission/submission.csv.
        debug (bool): If True, runs inference on a small subset of the test data.
                      Defaults to Config.DEBUG.
    """
    # Update Config debug flag to ensure get_dataloaders respects the requested mode
    if debug != Config.DEBUG:
        Config.DEBUG = debug

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # Load Data
    # get_dataloaders handles caching and reads Config.DEBUG internally
    print("Loading test data...")
    dataloaders = get_dataloaders()
    test_loader = dataloaders["test"]

    # Initialize Model
    model = ShallowCNN().to(device)

    # Load Weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights file not found at: {model_path}")

    print(f"Loading model weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Set to evaluation mode
    model.eval()

    predictions = []
    clip_names_list = []

    print("Generating predictions...")
    with torch.no_grad():
        for inputs, clip_names in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            # Model output is (Batch_Size, 1) with Sigmoid applied
            outputs = model(inputs)

            # Flatten to 1D array
            probs = outputs.cpu().numpy().flatten()

            predictions.extend(probs)
            clip_names_list.extend(clip_names)

    # Create Submission DataFrame
    df = pd.DataFrame({"clip": clip_names_list, "probability": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved successfully to {output_path}")
    print(f"Total predictions: {len(df)}")
