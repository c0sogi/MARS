import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import get_dataset
from library.model import RNAResidualBiGRU


def predict_and_format(device=None, batch_size=None):
    """
    Runs inference on the test set using the best trained model and generates
    a submission CSV file in the required format.

    Args:
        device (str, optional): Device to run inference on ('cpu' or 'cuda').
                                Defaults to Config.DEVICE.
        batch_size (int, optional): Batch size for inference.
                                    Defaults to Config.BATCH_SIZE.
    """
    # 1. Configuration and Setup
    Config.setup()

    if device is None:
        device = Config.DEVICE

    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    print(f"Starting inference on device: {device}")

    # 2. Load Test Data
    # We use load_cached_data=True to utilize pre-processed data if available
    test_dataset = get_dataset("test", load_cached_data=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    # 3. Load Model
    model = RNAResidualBiGRU().to(device)

    model_path = Config.BEST_MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Best model weights not found at {model_path}. Train the model first."
        )

    print(f"Loading model weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Inference Loop
    all_ids = []
    all_preds = []

    print("Running inference...")
    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to device
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            pair_dist = batch["pair_dist"].to(device)
            ids = batch["id"]

            # Forward pass: (Batch, SeqLen, 3)
            # Outputs correspond to: [reactivity, deg_Mg_pH10, deg_Mg_50C]
            preds = model(sequence, loop_type, pair_dist)

            # Move to CPU and store
            all_preds.append(preds.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate all predictions: Shape (N_samples, 107, 3)
    if len(all_preds) == 0:
        print("No predictions generated.")
        return

    predictions = np.concatenate(all_preds, axis=0)
    n_samples = predictions.shape[0]
    seq_len = predictions.shape[1]  # Should be 107

    print(
        f"Generated predictions for {n_samples} samples with sequence length {seq_len}."
    )

    # 5. Vectorized Formatting for Submission
    # We need to flatten the data to (N_samples * SeqLen) rows.

    # Create 'id_seqpos' column
    # Repeat IDs: [id1, id1, ..., id2, id2, ...]
    ids_repeated = np.repeat(all_ids, seq_len)

    # Tile positions: [0, 1, ..., 106, 0, 1, ..., 106]
    seqpos_tiled = np.tile(np.arange(seq_len), n_samples)

    # Vectorized string formatting for id_seqpos
    # Note: Using list comprehension here as numpy string operations can be tricky with mixed types
    id_seqpos_list = [f"{i}_{p}" for i, p in zip(ids_repeated, seqpos_tiled)]

    # Flatten predictions
    # predictions shape: (N, L, 3)
    # We flatten the first two dimensions to (N*L, 3)
    flat_preds = predictions.reshape(-1, 3)

    # Extract scored columns
    # Index 0: reactivity
    # Index 1: deg_Mg_pH10
    # Index 2: deg_Mg_50C
    reactivity = flat_preds[:, 0]
    deg_Mg_pH10 = flat_preds[:, 1]
    deg_Mg_50C = flat_preds[:, 2]

    # Create unscored columns (filled with zeros)
    zeros = np.zeros_like(reactivity)

    # Construct DataFrame
    submission_df = pd.DataFrame(
        {
            "id_seqpos": id_seqpos_list,
            "reactivity": reactivity,
            "deg_Mg_pH10": deg_Mg_pH10,
            "deg_pH10": zeros,  # Unscored
            "deg_Mg_50C": deg_Mg_50C,
            "deg_50C": zeros,  # Unscored
        }
    )

    # Ensure correct column order
    cols_order = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    submission_df = submission_df[cols_order]

    # 6. Save Submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Total rows: {len(submission_df)}")
    print("Sample check (first 5 rows):")
    print(submission_df.head())
