import torch
import pandas as pd
import os
import numpy as np
from library.config import Config
from library.model import AudioResNet18
from library.dataset import get_test_loader


def generate_submission(
    checkpoint_path=Config.MODEL_CHECKPOINT_PATH, output_path=Config.SUBMISSION_PATH
):
    """
    Generates predictions for the test set using the trained model and saves them to a CSV file.

    Args:
        checkpoint_path (str): Path to the saved model checkpoint.
        output_path (str): Path where the submission CSV will be saved.
    """
    device = torch.device(Config.DEVICE)
    print(f"Running inference on device: {device}")

    # 1. Initialize Model
    model = AudioResNet18(num_classes=Config.NUM_CLASSES)

    # 2. Load Weights
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. Please train the model first."
        )

    # Load state dict with map_location to handle CPU/GPU transfer if necessary
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # 3. Get Data Loader
    # Note: get_test_loader reads from Config.TEST_METADATA_PATH internally
    test_loader = get_test_loader()

    predictions = []

    # 4. Inference Loop
    print("Starting batch inference...")
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)

            # Get predicted class index
            _, predicted = torch.max(outputs, 1)

            # Move to CPU and store
            predictions.extend(predicted.cpu().numpy())

    # 5. Map Indices to Labels
    # Load metadata to ensure we match filenames correctly (order is preserved by DataLoader)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)
    fnames = df_test["fname"].tolist()

    if len(fnames) != len(predictions):
        print(
            f"Warning: Mismatch between metadata files ({len(fnames)}) and predictions ({len(predictions)})"
        )

    # Convert indices to string labels
    predicted_labels = [Config.IDX2LABEL[int(idx)] for idx in predictions]

    # 6. Create Submission DataFrame
    submission_df = pd.DataFrame({"fname": fnames, "label": predicted_labels})

    # 7. Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission successfully saved to {output_path}")
    print(submission_df.head())
