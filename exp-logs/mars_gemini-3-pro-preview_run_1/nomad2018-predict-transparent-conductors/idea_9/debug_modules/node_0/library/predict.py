import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader

from library.config import Config
from library.data import get_datasets, CollateFn
from library.model import SIRDS_SP
from library.utils import set_seed


def generate_submission(
    model_path=Config.MODEL_SAVE_PATH,
    output_path=Config.SUBMISSION_PATH,
    batch_size=Config.BATCH_SIZE,
    device=None,
):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model_path (str): Path to the trained model checkpoint.
        output_path (str): Path where the submission CSV will be saved.
        batch_size (int): Batch size for inference.
        device (torch.device, optional): Device to run inference on.
                                         If None, detects CUDA availability.
    """
    # 1. Setup
    set_seed(Config.SEED)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running inference on {device}...")

    # 2. Prepare Data
    # We use get_datasets to ensure the scaler is fitted on training data
    # and correctly applied to the test data.
    print("Loading datasets...")
    # load_cached_data=True attempts to load .npz files from working dir
    _, _, test_dataset = get_datasets(load_cached_data=True)

    collate_fn = CollateFn()
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Load Model
    print(f"Loading model from {model_path}...")
    model = SIRDS_SP()

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path}. Please train the model first."
        )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. Inference Loop
    all_ids = []
    all_preds = []

    print("Starting prediction loop...")
    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to device
            atom_features = batch["atom_features"].to(device)
            batch_indices = batch["batch_indices"].to(device)
            global_features = batch["global_features"].to(device)
            spacegroups = batch["spacegroups"].to(device)
            ids = batch["ids"]  # Keep on CPU for dataframe

            # Forward pass
            # Output shape: (B, 2) -> [formation_energy, bandgap_energy]
            preds = model(atom_features, batch_indices, global_features, spacegroups)

            # Collect results
            all_preds.append(preds.cpu().numpy())
            all_ids.append(ids.numpy())

    # 5. Post-processing
    # Concatenate all batches
    predictions = np.concatenate(all_preds, axis=0)
    ids = np.concatenate(all_ids, axis=0)

    # Inverse transform targets
    # Training used log1p, so we use expm1 to revert
    # predictions[:, 0] is formation_energy
    # predictions[:, 1] is bandgap_energy
    final_preds = np.expm1(predictions)

    # 6. Generate Submission DataFrame
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": final_preds[:, 0],
            "bandgap_energy_ev": final_preds[:, 1],
        }
    )

    # Sort by ID to ensure consistent order (though not strictly required by CSV format)
    submission_df.sort_values("id", inplace=True)

    # 7. Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
    print("Head of submission:")
    print(submission_df.head())
