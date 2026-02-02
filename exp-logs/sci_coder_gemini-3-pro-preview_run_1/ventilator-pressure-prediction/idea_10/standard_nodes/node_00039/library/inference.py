import os
import torch
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import seed_everything
from library.model import DeepSupervisedVentilatorModel
from library.dataset import get_dataloaders


def predict(load_cached_data: bool = True):
    """
    Runs inference on the test set using the trained model and generates a submission file.

    Args:
        load_cached_data (bool): Whether to attempt loading pre-processed data from cache.
                                 Defaults to True.

    Returns:
        pd.DataFrame: The submission dataframe containing 'id' and 'pressure'.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Inference using device: {device}")

    # 2. Load Data
    # We only need the test_loader and test_ids.
    # get_dataloaders handles the feature engineering pipeline and caching.
    print("Loading test data...")
    _, _, test_loader, test_ids = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=load_cached_data,
    )

    # 3. Load Model
    print(f"Loading model from {Config.MODEL_PATH}...")
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_PATH}. Please train the model first."
        )

    model = DeepSupervisedVentilatorModel().to(device)

    # Load state dict
    state_dict = torch.load(Config.MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)

    # Set to evaluation mode (disables dropout, returns only final head output)
    model.eval()

    # 4. Inference Loop
    print("Running inference...")
    all_preds = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            # Move data to device
            data = data.to(device, non_blocking=True)

            # Forward pass
            # In eval mode, model returns tensor of shape (Batch, Seq_Len)
            preds = model(data)

            # Move to CPU and store
            all_preds.append(preds.cpu().numpy())

    # 5. Post-processing
    # Concatenate all batches: Result is (N_breaths, Seq_Len)
    predictions_matrix = np.concatenate(all_preds, axis=0)

    # Flatten to (N_breaths * Seq_Len,) to match the flat test_ids
    flat_predictions = predictions_matrix.flatten()

    # Verify shapes
    if len(flat_predictions) != len(test_ids):
        raise ValueError(
            f"Shape mismatch: Predictions ({len(flat_predictions)}) vs IDs ({len(test_ids)}). "
            "Ensure test set batching and reshaping are consistent."
        )

    # 6. Generate Submission
    print("Generating submission dataframe...")
    submission_df = pd.DataFrame({"id": test_ids, "pressure": flat_predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save to CSV
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print("Inference complete.")
    return submission_df
