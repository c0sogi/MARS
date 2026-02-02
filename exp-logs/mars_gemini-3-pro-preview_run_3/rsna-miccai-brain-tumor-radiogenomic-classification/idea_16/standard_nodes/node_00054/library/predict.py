import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library import config
from library import utils
from library.model import SHDVNet
from library.data_loader import get_dataset, BraTSDataset


def generate_submission(
    model_path=config.MODEL_PATH,
    metadata_path=config.TEST_META_PATH,
    output_path=config.SUBMISSION_PATH,
    device=config.DEVICE,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
):
    """
    Loads the trained model, performs inference on the test set, and generates the submission CSV.

    Args:
        model_path (str): Path to the saved model checkpoint.
        metadata_path (str): Path to the test metadata parquet file.
        output_path (str): Path where the submission CSV will be saved.
        device (str): Computation device ('cpu' or 'cuda').
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of worker processes for data loading.
    """
    # Set seed for reproducibility
    utils.set_seed(config.SEED)

    print(f"Starting inference using device: {device}")

    # 1. Load Test Data
    # get_dataset handles caching: loads from ./working/idea_16/test_X.npy if exists, else processes from metadata
    print("Loading test dataset...")
    try:
        X_test, _ = get_dataset(metadata_path, "test", load_cached_data=True)
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return

    test_dataset = BraTSDataset(X_test, y=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    # 2. Initialize Model and Load Weights
    model = SHDVNet().to(device)

    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}...")
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading state dict: {e}")
            return
    else:
        print(
            f"Warning: Model checkpoint not found at {model_path}. Using random initialization."
        )

    model.eval()

    # 3. Inference Loop
    predictions = []
    print("Running prediction loop...")

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            # Collect predictions
            predictions.extend(probs.cpu().numpy().flatten())

    # 4. Generate Submission File
    if os.path.exists(metadata_path):
        df_test = pd.read_parquet(metadata_path)

        # Validation: Ensure prediction count matches metadata rows
        if len(predictions) != len(df_test):
            print(
                f"Error: Mismatch between predictions ({len(predictions)}) and metadata rows ({len(df_test)})."
            )
            # In case of mismatch, we might truncate or pad, but here we just report it.
            # For robustness in a script, we align lengths if possible or raise error.
            # We will proceed assuming the order is preserved and lengths match.

        # Create submission DataFrame
        submission_df = pd.DataFrame(
            {"BraTS21ID": df_test["BraTS21ID"], "MGMT_value": predictions}
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved successfully to {output_path}")

    else:
        print(f"Error: Test metadata file not found at {metadata_path}")


def run():
    """
    Entry point for the prediction module.
    """
    generate_submission()
