import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.model import CuratedIdentityNet
from library.dataset import VentilatorDataset, _process_and_cache


def predict(
    model_path: str = Config.MODEL_PATH,
    output_path: str = Config.SUBMISSION_PATH,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
    device: str = Config.DEVICE,
    load_cached_data: bool = True,
):
    """
    Runs inference on the test set using the trained model and generates a submission file.

    Args:
        model_path (str): Path to the trained model checkpoint.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of worker threads for data loading.
        device (str): Device to run inference on ('cpu' or 'cuda').
        load_cached_data (bool): Whether to use cached preprocessed data.
    """
    # 1. Setup Environment
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device(device)

    # 2. Prepare Test Data
    # Directly access the processing function to avoid overhead of loading train/val data
    print("Preparing test dataset...")
    x_test, static_test, u_out_test, _, test_ids = _process_and_cache(
        "test", load_cached_data=load_cached_data
    )

    # Create Dataset and DataLoader
    test_dataset = VentilatorDataset(x_test, static_test, u_out_test, None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 3. Load Model
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = CuratedIdentityNet()
    model.to(device)

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 4. Inference Loop
    print("Running inference...")
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            static = batch["static"].to(device)

            # Forward pass
            # Model returns (final_pred, aux_pred)
            final_pred, _ = model(x, static)

            # Squeeze the last dimension (Batch, Seq_Len, 1) -> (Batch, Seq_Len)
            final_pred = final_pred.squeeze(-1)

            # Move to CPU and collect
            predictions.append(final_pred.cpu().numpy())

    # 5. Post-processing
    # Concatenate all batches: (N_test_breaths, Seq_Len)
    predictions = np.concatenate(predictions, axis=0)

    # Flatten predictions to 1D array to match the submission format (id-wise)
    predictions_flat = predictions.flatten()

    # Ensure test_ids are also flat
    test_ids_flat = test_ids.flatten()

    if len(predictions_flat) != len(test_ids_flat):
        raise ValueError(
            f"Shape mismatch: Preds {len(predictions_flat)} vs IDs {len(test_ids_flat)}"
        )

    # 6. Save Submission
    print(f"Saving submission to {output_path}...")
    submission_df = pd.DataFrame({"id": test_ids_flat, "pressure": predictions_flat})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print("Submission generation complete.")
