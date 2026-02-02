import os
import numpy as np
import pandas as pd
import torch
from library.config import (
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    SEED,
)
from library.model import GPIMSDS
from library.data_processing import get_dataloaders

# Set random seeds for reproducibility
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


def run_inference(load_cached_data=True):
    """
    Loads the trained GPIMSDS model and generates predictions for the test set.

    Args:
        load_cached_data (bool): Whether to load pre-processed data and scalers from cache.
                                 If False, data processing will be re-run.

    Returns:
        pd.DataFrame: The submission dataframe containing IDs and predictions.
    """
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference using device: {device}")

    # 2. Load Data
    # get_dataloaders handles caching and scaling. We only need the test_loader.
    # It ensures that if cached scalers exist (from training), they are used.
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data)

    # 3. Load Model
    print(f"Loading model architecture and weights from {MODEL_SAVE_PATH}...")
    # Initialize model using hyperparameters from config (embedded in class init)
    model = GPIMSDS().to(device)

    if os.path.exists(MODEL_SAVE_PATH):
        state_dict = torch.load(MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(
            f"Model checkpoint not found at {MODEL_SAVE_PATH}. Please train the model before running inference."
        )

    model.eval()

    # 4. Prediction Loop
    predictions = []
    ids = []

    print("Running prediction loop on test set...")
    with torch.no_grad():
        for batch in test_loader:
            # Move batch data to device
            atom_feats = batch["atom_features"].to(device)
            global_feats = batch["global_features"].to(device)
            batch_idx = batch["batch_index"].to(device)
            batch_ids = batch["ids"].numpy()

            # Forward pass
            outputs = model(atom_feats, global_feats, batch_idx)

            # Inverse transform targets
            # The model predicts log(1 + y), so we apply exp(x) - 1
            preds = torch.expm1(outputs).cpu().numpy()

            # Enforce physical constraint: Energies cannot be negative
            preds = np.maximum(preds, 0.0)

            predictions.append(preds)
            ids.append(batch_ids)

    # Concatenate results from all batches
    predictions = np.concatenate(predictions, axis=0)
    ids = np.concatenate(ids, axis=0)

    # 5. Create Submission File
    sub_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    # Sort by ID as required by the submission format
    sub_df = sub_df.sort_values("id")

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # Save submission
    sub_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print("Head of submission file:")
    print(sub_df.head())

    return sub_df
