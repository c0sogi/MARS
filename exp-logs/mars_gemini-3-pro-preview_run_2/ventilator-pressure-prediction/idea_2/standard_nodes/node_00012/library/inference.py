import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.data_loader import get_data_loaders
from library.model import HybridLSTMTransformer


def predict_test_set(model, test_loader, device):
    """
    Runs inference on the test set using the provided model.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run inference on.

    Returns:
        tuple: (ids, predictions) as flattened numpy arrays.
    """
    model.eval()
    all_ids = []
    all_preds = []

    print("Starting inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            cont_x = batch["cont"].to(device)
            cat_x = batch["cat"].to(device)

            # IDs are needed for submission mapping
            ids = batch["ids"].numpy()  # Shape: (Batch, Seq)

            # Forward pass
            # Output shape: (Batch, Seq)
            preds = model(cont_x, cat_x)

            # Flatten sequences immediately to save memory and prepare for CSV
            all_ids.append(ids.flatten())
            all_preds.append(preds.cpu().numpy().flatten())

    # Concatenate all batches
    final_ids = np.concatenate(all_ids)
    final_preds = np.concatenate(all_preds)

    return final_ids, final_preds


def create_submission_file(ids, preds, save_path):
    """
    Creates a submission DataFrame and saves it to a CSV file.

    Args:
        ids (np.array): Flattened array of time step IDs.
        preds (np.array): Flattened array of predicted pressures.
        save_path (str): Path to save the CSV file.
    """
    print(f"Creating submission file with {len(ids)} predictions...")

    # Create DataFrame
    df_sub = pd.DataFrame({"id": ids, "pressure": preds})

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save to CSV
    df_sub.to_csv(save_path, index=False)
    print(f"Submission saved successfully to {save_path}")


def run_inference(load_cached_data=True, debug=False):
    """
    Main function to orchestrate the inference pipeline.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
        debug (bool): Whether to run in debug mode (subset of data).
    """
    # 1. Setup Device
    device = torch.device(Config.DEVICE)
    print(f"Running inference on device: {device}")

    # 2. Load Data
    # We only strictly need the test loader here, but get_data_loaders returns all three
    _, _, test_loader = get_data_loaders(load_cached_data=load_cached_data, debug=debug)

    # 3. Initialize Model
    model = HybridLSTMTransformer().to(device)

    # 4. Load Weights
    model_path = Config.BEST_MODEL_PATH
    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Model checkpoint not found at {model_path}.")
        print("Inference will run with initialized (random) weights.")

    # 5. Predict
    ids, preds = predict_test_set(model, test_loader, device)

    # 6. Save Submission
    create_submission_file(ids, preds, Config.SUBMISSION_PATH)
