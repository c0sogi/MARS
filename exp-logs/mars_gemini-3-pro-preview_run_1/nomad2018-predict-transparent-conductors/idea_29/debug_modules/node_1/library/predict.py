import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import (
    TEST_CSV,
    WORKING_DIR,
    SUBMISSION_FILE,
    BATCH_SIZE,
    ATOMIC_HIDDEN_DIM,
    GLOBAL_HIDDEN_DIM,
    FUSION_HIDDEN_DIM,
    DROPOUT,
)
from library.dataset import MaterialDataset, collate_materials
from library.model import PIGWDS


def generate_predictions(
    batch_size=BATCH_SIZE,
    load_cached_data=True,
    model_path="best_model.pt",
    scalers_path="scalers.npz",
):
    """
    Generates predictions for the test set using the trained PIG-WDS model.

    Args:
        batch_size (int): Batch size for inference.
        load_cached_data (bool): Whether to load pre-processed data from cache.
        model_path (str): Name of the model file in WORKING_DIR.
        scalers_path (str): Name of the scalers file in WORKING_DIR.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Scalers
    full_scalers_path = os.path.join(WORKING_DIR, scalers_path)
    if not os.path.exists(full_scalers_path):
        raise FileNotFoundError(f"Scalers file not found at {full_scalers_path}")

    print(f"Loading scalers from {full_scalers_path}...")
    # Load NpzFile and convert to dict
    with np.load(full_scalers_path) as data:
        scalers = {key: data[key] for key in data.files}

    # 2. Initialize Test Dataset
    print("Initializing Test Dataset...")
    test_dataset = MaterialDataset(
        metadata_path=TEST_CSV, scalers=scalers, load_cached_data=load_cached_data
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_materials,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 3. Model Initialization
    # Infer input dimensions from the first sample
    sample = test_dataset[0]
    atomic_input_dim = sample["atomic_features"].shape[1]
    global_input_dim = sample["global_features"].shape[0]
    # Output dim is fixed to 2 for this task (formation energy, bandgap)
    output_dim = 2

    print(f"Atomic Input Dim: {atomic_input_dim}")
    print(f"Global Input Dim: {global_input_dim}")

    model = PIGWDS(
        atomic_input_dim=atomic_input_dim,
        global_input_dim=global_input_dim,
        atomic_hidden_dim=ATOMIC_HIDDEN_DIM,
        global_hidden_dim=GLOBAL_HIDDEN_DIM,
        fusion_hidden_dim=FUSION_HIDDEN_DIM,
        output_dim=output_dim,
        dropout=DROPOUT,
    ).to(device)

    # 4. Load Model Weights
    full_model_path = os.path.join(WORKING_DIR, model_path)
    if not os.path.exists(full_model_path):
        raise FileNotFoundError(f"Model file not found at {full_model_path}")

    print(f"Loading model weights from {full_model_path}...")
    model.load_state_dict(torch.load(full_model_path, map_location=device))
    model.eval()

    # 5. Inference Loop
    print("Starting inference...")
    ids_all = []
    predictions_all = []

    with torch.no_grad():
        for batch in test_loader:
            atomic_feats = batch["atomic_features"].to(device)
            global_feats = batch["global_features"].to(device)
            mask = batch["mask"].to(device)

            # Forward pass
            preds = model(atomic_feats, global_feats, mask)

            # Store results
            ids_all.extend(batch["ids"])
            predictions_all.append(preds.cpu().numpy())

    # Concatenate predictions
    predictions_all = np.concatenate(predictions_all, axis=0)

    # 6. Inverse Transformation
    # Training used log1p, so we use expm1 to revert
    print("Applying inverse transformation (expm1)...")
    original_scale_preds = np.expm1(predictions_all)

    # 7. Create Submission DataFrame
    submission_df = pd.DataFrame(
        {
            "id": ids_all,
            "formation_energy_ev_natom": original_scale_preds[:, 0],
            "bandgap_energy_ev": original_scale_preds[:, 1],
        }
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(SUBMISSION_FILE), exist_ok=True)

    # Save submission
    print(f"Saving submission to {SUBMISSION_FILE}...")
    submission_df.to_csv(SUBMISSION_FILE, index=False)
    print("Submission generated successfully.")
    print(submission_df.head())
