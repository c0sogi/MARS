import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import get_datasets, collate_fn
from library.model import APDeepSets


def generate_submission(debug=False, max_samples=None):
    """
    Generates the submission file for the competition.

    This function:
    1. Loads the test dataset (using cached/processed features).
    2. Loads the trained AP-DeepSets model.
    3. Runs inference in batches.
    4. Applies the inverse transformation (exp(x) - 1) to the predictions.
    5. Saves the results to a CSV file.

    Args:
        debug (bool): If True, runs inference on a smaller subset of the test data for debugging.
        max_samples (int): Number of samples to use if debug is True.
    """
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Test Dataset
    # get_datasets handles loading from metadata, feature extraction, caching, and normalization.
    # We ignore train and val datasets here.
    print("Loading test dataset...")
    _, _, test_dataset = get_datasets(debug=debug, max_samples=max_samples)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 3. Load Model
    print(f"Loading model from {Config.MODEL_PATH}...")
    model = APDeepSets()

    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_PATH}. Please train the model first."
        )

    # Load weights
    state_dict = torch.load(Config.MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. Inference
    print("Running inference...")
    all_ids = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            # batch keys: 'ids', 'global_features', 'atomic_features', 'batch_indices', 'targets'
            global_features = batch["global_features"].to(device)
            atomic_features = batch["atomic_features"].to(device)
            batch_indices = batch["batch_indices"].to(device)

            # Store IDs for submission mapping
            all_ids.extend(batch["ids"])

            # Prepare input dictionary for the model
            model_input = {
                "global_features": global_features,
                "atomic_features": atomic_features,
                "batch_indices": batch_indices,
            }

            # Forward pass
            outputs = model(model_input)

            # Collect predictions
            all_preds.append(outputs.cpu().numpy())

    # 5. Post-processing
    # Concatenate all batches
    predictions = np.concatenate(all_preds, axis=0)

    # Inverse transformation
    # The model was trained on log1p(targets), so we apply expm1 to get back to original scale
    # formation_energy_ev_natom and bandgap_energy_ev
    original_scale_preds = np.expm1(predictions)

    # 6. Create Submission DataFrame
    # Ensure columns match the sample submission format
    submission_df = pd.DataFrame(
        {
            "id": all_ids,
            "formation_energy_ev_natom": original_scale_preds[:, 0],
            "bandgap_energy_ev": original_scale_preds[:, 1],
        }
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("First 5 rows of submission:")
    print(submission_df.head())
