import os
import torch
import pandas as pd
from torch.utils.data import Subset
from library.config import Config
from library.utils import set_seed
from library.model import SAMPNet
from library.data_loader import get_data_loaders
from library.train import decode_predictions


def generate_predictions(debug=False):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        debug (bool): If True, runs on a small subset of the data for debugging purposes.
    """
    # 1. Setup Environment
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Load Data
    # We only need the test loader here.
    # get_data_loaders handles stats loading internally.
    _, _, test_loader = get_data_loaders(debug=debug)

    # 3. Load Model
    model = SAMPNet().to(device)
    checkpoint_path = Config.BEST_MODEL_PATH

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        # This case should ideally not happen in the final run if training succeeded
        print(
            f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights."
        )

    model.eval()

    # 4. Inference Loop
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            skel, audio, _, _, lengths = batch

            # Ensure data is valid (collate_fn filters None, but check for safety)
            if skel is None:
                continue

            # Move inputs to device
            skel = skel.to(device)
            audio = audio.to(device)

            # Forward pass
            logits, _ = model(skel, audio, lengths)

            # Decode predictions
            # Uses the logic from library.train: Median Filter -> RLE -> Filter Short/Background
            batch_preds = decode_predictions(logits, lengths)
            all_preds.extend(batch_preds)

    # 5. Align Predictions with Sample IDs
    # The DataLoader preserves order (shuffle=False).
    # We need to extract sample_ids from the underlying dataset.
    dataset = test_loader.dataset

    if isinstance(dataset, Subset):
        # If debug=True, we are working with a Subset.
        # Map subset indices back to the original dataframe.
        indices = dataset.indices
        sample_ids = dataset.dataset.df.iloc[indices]["sample_id"].tolist()
    else:
        # Standard dataset
        sample_ids = dataset.df["sample_id"].tolist()

    # Verify alignment
    if len(sample_ids) != len(all_preds):
        # This might happen if specific batches were dropped entirely due to data corruption,
        # but the loader logic usually replaces corrupt files with zeros or filters them.
        # Given the provided loader filters in __init__, lengths should match.
        print(
            f"Error: Mismatch between Sample IDs ({len(sample_ids)}) and Predictions ({len(all_preds)})."
        )
        return

    # 6. Write Submission CSV
    output_path = Config.SUBMISSION_PATH

    with open(output_path, "w") as f:
        for sid, preds in zip(sample_ids, all_preds):
            # preds is a list of integers (e.g., [2, 12, 3])
            # Join with commas
            pred_str = ",".join(map(str, preds))

            # Format: SessionID,pred1,pred2,...
            line = f"{sid},{pred_str}"
            f.write(line + "\n")

    print(f"Submission saved to {output_path}")
