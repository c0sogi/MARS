import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import get_device
from library.dataset import get_dataloaders
from library.model import DP_GI_BiLSTM


def generate_predictions(model, loader, device):
    """
    Generates predictions for the test set using the provided model.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): The test data loader.
        device (torch.device): The device to run inference on.

    Returns:
        np.ndarray: Flattened array of predictions.
    """
    model.eval()
    predictions = []

    # Ensure no gradients are calculated to save memory and computation
    with torch.no_grad():
        for X in loader:
            X = X.to(device)
            # Forward pass
            pred = model(X)
            # Flatten predictions and move to CPU
            predictions.append(pred.cpu().numpy().flatten())

    return np.concatenate(predictions)


def create_submission(predictions, test_ids, output_path):
    """
    Formats the predictions into a submission DataFrame and saves it to CSV.

    Args:
        predictions (np.ndarray): The predicted pressure values.
        test_ids (np.ndarray): The corresponding IDs.
        output_path (str): Path to save the submission CSV.
    """
    # Validate alignment
    if len(test_ids) != len(predictions):
        print(
            f"Warning: Length mismatch. IDs: {len(test_ids)}, Preds: {len(predictions)}"
        )

    submission_df = pd.DataFrame({"id": test_ids, "pressure": predictions})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_inference(config=Config):
    """
    Orchestrates the full inference pipeline:
    1. Loads data (and updates Config.INPUT_DIM).
    2. Initializes model.
    3. Loads best weights.
    4. Generates predictions.
    5. Saves submission.
    """
    device = get_device()
    print(f"Inference using device: {device}")

    # 1. Load Data
    # calling get_dataloaders is critical here because it triggers the
    # feature engineering pipeline which calculates the correct Config.INPUT_DIM
    print("Loading test data...")
    _, _, test_loader, test_ids = get_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        load_cached_data=True,
    )

    # 2. Initialize Model
    # Config.INPUT_DIM is now updated
    print(f"Initializing model with Input Dim: {config.INPUT_DIM}...")
    model = DP_GI_BiLSTM(config).to(device)

    # 3. Load Weights
    checkpoint_path = config.MODEL_CHECKPOINT_PATH
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    print(f"Loading weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    # 4. Generate Predictions
    print("Generating predictions...")
    preds = generate_predictions(model, test_loader, device)

    # 5. Create Submission
    print("Creating submission file...")
    create_submission(preds, test_ids, config.SUBMISSION_PATH)
