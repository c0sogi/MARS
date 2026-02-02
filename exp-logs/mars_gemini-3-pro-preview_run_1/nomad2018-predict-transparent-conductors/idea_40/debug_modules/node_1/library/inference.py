import torch
import numpy as np
import pandas as pd
import os
from library.config import Config
from library.model import DC3_WDS
from library.data_loader import get_loaders


def predict(model, dataloader, device):
    """
    Runs the model on the provided dataloader and returns raw predictions and IDs.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for the test set.
        device: Torch device.

    Returns:
        predictions (np.ndarray): Raw model outputs (log scale).
        ids (np.ndarray): Corresponding sample IDs.
    """
    model.eval()
    predictions = []
    ids = []

    print("Running inference...")

    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            atomic_feats = batch["atomic_feats"].to(device)
            global_feats = batch["global_feats"].to(device)
            mask = batch["mask"].to(device)
            batch_ids = batch["id"].cpu().numpy()

            # Forward pass
            outputs = model(atomic_feats, global_feats, mask)

            # Collect results
            predictions.append(outputs.cpu().numpy())
            ids.append(batch_ids)

    predictions = np.concatenate(predictions, axis=0)
    ids = np.concatenate(ids, axis=0)

    return predictions, ids


def inverse_transform(predictions):
    """
    Converts log-scale predictions back to the original energy scale.
    Applies expm1 (exp(x) - 1).

    Args:
        predictions (np.ndarray): Log-scale predictions.

    Returns:
        np.ndarray: Predictions in original scale.
    """
    return np.expm1(predictions)


def generate_submission(predictions, ids, output_path):
    """
    Formats the predictions and saves them to a CSV file.

    Args:
        predictions (np.ndarray): Predictions in original scale.
        ids (np.ndarray): Sample IDs.
        output_path (str): Path to save the CSV.
    """
    # Create DataFrame
    df = pd.DataFrame(predictions, columns=Config.TARGET_COLS)
    df.insert(0, "id", ids)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_inference(load_cached_data=True):
    """
    Main entry point for inference. Loads data, model, generates predictions, and saves submission.
    """
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Load Data
    # We only need the test loader, but get_loaders returns all three
    _, _, test_loader = get_loaders(load_cached_data=load_cached_data)

    # 2. Initialize Model
    model = DC3_WDS().to(device)

    # 3. Load Trained Weights
    if os.path.exists(Config.MODEL_PATH):
        print(f"Loading model weights from {Config.MODEL_PATH}")
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    else:
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_PATH}. Please train the model first."
        )

    # 4. Predict
    raw_preds, ids = predict(model, test_loader, device)

    # 5. Inverse Transform
    final_preds = inverse_transform(raw_preds)

    # 6. Generate Submission
    generate_submission(final_preds, ids, Config.SUBMISSION_PATH)
