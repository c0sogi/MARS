import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

from library.config import Config
from library.data import get_datasets, collate_fn
from library.model import RBFDualStreamDeepSets


def set_seed(seed=42):
    """Sets random seed for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_predictions(
    model_path=Config.MODEL_SAVE_PATH,
    batch_size=Config.BATCH_SIZE,
    load_cached_data=True,
    num_workers=2,
    limit_samples=None,
):
    """
    Generates predictions for the test set using the trained model.

    Args:
        model_path (str): Path to the saved model weights.
        batch_size (int): Batch size for inference.
        load_cached_data (bool): Whether to load pre-processed data from cache.
        num_workers (int): Number of worker threads for DataLoader.
        limit_samples (int, optional): Limit the number of test samples for debugging.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    # We call get_datasets to ensure the scaler is fitted on the training set
    # and applied consistently to the test set.
    print("Loading datasets...")
    # We ignore train and val datasets here, but they are needed to fit the scaler internally
    _, _, test_dataset = get_datasets(load_cached_data=load_cached_data)

    if limit_samples is not None:
        print(f"Limiting test set to {limit_samples} samples.")
        indices = list(range(min(len(test_dataset), limit_samples)))
        test_dataset = Subset(test_dataset, indices)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 2. Load Model
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = RBFDualStreamDeepSets().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Inference
    ids = []
    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            atomic_features = batch["atomic_features"].to(device)
            lattice_features = batch["lattice_features"].to(device)
            batch_indices = batch["batch_indices"].to(device)

            # Forward pass
            outputs = model(atomic_features, lattice_features, batch_indices)

            # Inverse transform: y = exp(y') - 1
            # The model was trained on log1p(targets)
            preds = torch.expm1(outputs)

            # Clamp to ensure non-negative values (energies >= 0)
            preds = torch.clamp(preds, min=0.0)

            ids.extend(batch["ids"].numpy())
            predictions.extend(preds.cpu().numpy())

    # 4. Save Submission
    predictions = np.array(predictions)
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    # Ensure correct column order
    submission_df = submission_df[
        ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
    ]

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Preview
    print(submission_df.head())
