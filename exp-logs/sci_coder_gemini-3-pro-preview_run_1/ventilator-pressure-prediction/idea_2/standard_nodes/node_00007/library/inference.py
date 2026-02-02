import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.data_processing import DataProcessor
from library.dataset import VentilatorDataset
from library.model import HybridCNNLSTM


def generate_predictions(
    model_path: str = Config.MODEL_PATH,
    output_path: str = Config.SUBMISSION_PATH,
    sample_submission_path: str = Config.SAMPLE_SUBMISSION_PATH,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
    device: str = Config.DEVICE,
    load_cached_data: bool = True,
):
    """
    Generates predictions for the test set using a trained model and saves the submission file.

    Args:
        model_path (str): Path to the saved model weights (.pth file).
        output_path (str): Path where the submission CSV will be saved.
        sample_submission_path (str): Path to the sample submission file to ensure correct ID mapping.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of worker threads for DataLoader.
        device (str): Device to run inference on ('cpu' or 'cuda').
        load_cached_data (bool): Whether to attempt loading pre-processed data from cache.
    """
    # 1. Setup and Reproducibility
    seed_everything(Config.SEED)
    device = torch.device(device)
    print(f"Starting inference on device: {device}")

    # 2. Data Preparation
    # Initialize DataProcessor which handles caching and feature engineering
    processor = DataProcessor()

    print("Loading and processing test data...")
    # load_dataset returns (X, u_out) for the test split
    X_test, u_out_test = processor.load_dataset(
        split="test", load_cached_data=load_cached_data
    )

    # Create Dataset and DataLoader
    test_dataset = VentilatorDataset(X_test, u_out_test, y=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 3. Model Initialization and Loading
    print(f"Loading model from {model_path}...")
    model = HybridCNNLSTM().to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please train the model first."
        )

    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Inference Loop
    all_preds = []
    print("Generating predictions...")

    with torch.no_grad():
        for batch_idx, (X, _) in enumerate(test_loader):
            X = X.to(device)

            # Forward pass: returns (Batch, Seq_Len)
            preds = model(X)

            # Move to CPU and numpy
            all_preds.append(preds.cpu().numpy())

    # 5. Post-processing
    # Concatenate all batches: Shape (N_breaths, Seq_Len)
    predictions = np.concatenate(all_preds, axis=0)

    # Flatten the predictions to match the row-wise format of the submission file
    # The DataProcessor sorts data by breath_id and time_step, matching the submission structure
    predictions_flat = predictions.flatten()

    # 6. Submission Generation
    print(f"Reading sample submission from {sample_submission_path}...")
    sub_df = pd.read_csv(sample_submission_path)

    # Validation check for length consistency
    if len(sub_df) != len(predictions_flat):
        print(
            f"Warning: Length mismatch! Sample submission: {len(sub_df)}, Predictions: {len(predictions_flat)}"
        )
        # Handle mismatch by truncating to the smaller length to ensure file can be saved
        min_len = min(len(sub_df), len(predictions_flat))
        sub_df = sub_df.iloc[:min_len]
        predictions_flat = predictions_flat[:min_len]

    # Assign predictions
    sub_df[Config.COL_PRESSURE] = predictions_flat

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    print(f"Saving submission to {output_path}...")
    sub_df.to_csv(output_path, index=False)
    print("Submission generation complete.")
