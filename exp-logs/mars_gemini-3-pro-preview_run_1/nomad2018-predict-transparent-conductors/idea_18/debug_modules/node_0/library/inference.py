import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.data import CrystalDataset, collate_crystals
from library.model import CRNDSModel


def set_seed(seed):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict(load_cached_data=True, batch_size=Config.BATCH_SIZE, device=None):
    """
    Runs inference on the test dataset using the best trained model.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
        batch_size (int): Batch size for the DataLoader.
        device (torch.device, optional): Device to run inference on.

    Returns:
        tuple: (predictions, ids)
            - predictions (np.ndarray): Raw model outputs (log scale), shape (N, 2).
            - ids (list): List of crystal IDs corresponding to predictions.
    """
    if device is None:
        device = torch.device(Config.DEVICE)

    # Initialize Test Dataset
    # This will load scalers generated during training to ensure consistent normalization
    test_dataset = CrystalDataset(
        metadata_path=Config.TEST_META_PATH,
        cache_path=Config.TEST_CACHE_PATH,
        scalers_path=Config.SCALERS_CACHE_PATH,
        split="test",
        load_cached_data=load_cached_data,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_crystals,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Load Model
    print(f"Loading model from {Config.MODEL_CHECKPOINT_PATH}...")
    model = CRNDSModel().to(device)

    if not os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_CHECKPOINT_PATH}. "
            "Please run training first."
        )

    checkpoint = torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    all_predictions = []
    all_ids = []

    print("Running inference...")
    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            atomic_features = batch["atomic_features"].to(device)
            global_features = batch["global_features"].to(device)
            batch_index = batch["batch_index"].to(device)
            batch_ids = batch["ids"]

            # Forward pass
            outputs = model(atomic_features, global_features, batch_index)

            # Collect results
            all_predictions.append(outputs.cpu().numpy())
            all_ids.extend(batch_ids)

    return np.vstack(all_predictions), all_ids


def generate_submission(load_cached_data=True, batch_size=Config.BATCH_SIZE):
    """
    Generates the submission file by running inference and applying the inverse transformation.

    Args:
        load_cached_data (bool): Whether to use cached pre-processed data.
        batch_size (int): Batch size for inference.
    """
    # Ensure directories exist
    Config.setup()
    set_seed(Config.SEED)

    # 1. Predict (returns log-scale predictions)
    log_preds, ids = predict(load_cached_data=load_cached_data, batch_size=batch_size)

    # 2. Inverse Transformation
    # The model was trained on log(1+y), so we apply exp(x) - 1 to get original scale
    original_preds = np.expm1(log_preds)

    # 3. Create DataFrame
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": original_preds[:, 0],
            "bandgap_energy_ev": original_preds[:, 1],
        }
    )

    # 4. Sort by ID (required for consistent submission format)
    submission_df.sort_values("id", inplace=True)

    # 5. Save to CSV
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
