import os
import torch
import pandas as pd
import numpy as np
from typing import Optional

from library.config import Config
from library.model import VentilatorModel
from library.dataset import get_data_loaders
from library.utils import seed_everything


def generate_submission(
    model_filename: str = "model.pth",
    load_cached_data: bool = True,
    debug_limit: Optional[int] = None,
):
    """
    Loads the trained model, performs inference on the test set, and generates
    a submission CSV file.

    Args:
        model_filename (str): Name of the model file to load from the working directory.
        load_cached_data (bool): Whether to use cached preprocessed data.
        debug_limit (int, optional): Limit the number of samples for debugging.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Inference Device: {device}")

    model_path = os.path.join(Config.WORKING_DIR, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # 2. Data Loading
    # We use get_data_loaders to ensure the scaler is correctly loaded/fitted
    # and the test data is processed identically to training.
    print("Loading test data...")
    _, _, test_loader = get_data_loaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=load_cached_data,
        debug_limit=debug_limit,
    )

    # 3. Model Loading
    print(f"Loading model from {model_path}...")
    model = VentilatorModel(config=Config).to(device)

    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()

    # 4. Inference
    print("Starting inference...")
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for X, _, ids in test_loader:
            X = X.to(device)

            # Forward pass
            # Model returns (final_pred, aux_pred). We only need final_pred.
            final_pred, _ = model(X)

            # Move to CPU and store
            all_preds.append(final_pred.cpu().numpy())
            all_ids.append(ids.numpy())

    # 5. Post-processing
    # Concatenate all batches: (N_batches, Batch_Size, 80) -> (Total_Samples, 80)
    # Then flatten to (Total_Samples * 80)
    flat_preds = np.concatenate(all_preds, axis=0).flatten()
    flat_ids = np.concatenate(all_ids, axis=0).flatten()

    # 6. Create Submission DataFrame
    submission_df = pd.DataFrame({Config.ID: flat_ids, Config.PRESSURE: flat_preds})

    # Sort by ID just to be safe (though data loader should preserve order if not shuffled)
    submission_df = submission_df.sort_values(by=Config.ID)

    # 7. Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    save_path = Config.SUBMISSION_PATH

    submission_df.to_csv(save_path, index=False)

    print(f"Submission generated successfully.")
    print(f"Shape: {submission_df.shape}")
    print(f"Saved to: {save_path}")
