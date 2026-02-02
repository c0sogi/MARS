import os
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.data import get_dataloaders
from library.model import GEHN
from library.utils import seed_everything


def generate_submission(config=None, load_cached_data=True):
    """
    Generates the submission file for the RNA degradation prediction task.

    Args:
        config (Config, optional): Configuration object. If None, a new Config is initialized.
        load_cached_data (bool): Whether to load pre-processed data from cache.
                                 Defaults to True.
    """
    # 1. Setup
    seed_everything(42)
    if config is None:
        config = Config()

    device = config.device
    print(f"Generating submission on device: {device}")

    # 2. Data Loading
    # We only need the test loader
    _, _, test_loader = get_dataloaders(config, load_cached_data=load_cached_data)

    # 3. Model Loading
    model = GEHN(config).to(device)

    if not os.path.exists(config.best_model_path):
        raise FileNotFoundError(
            f"Model file not found at {config.best_model_path}. Please train the model first."
        )

    state_dict = torch.load(config.best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Inference
    all_preds = []
    all_ids = []

    print("Running inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            adj = batch["adj"].to(device)
            ids = batch["id"]  # List or array of strings

            # Forward pass
            # Output shape: (Batch, Seq_Len=107, Num_Targets=5)
            preds = model(inputs, adj)

            # Move to CPU and numpy
            preds_np = preds.cpu().numpy()

            all_preds.append(preds_np)
            all_ids.extend(ids)

    # Concatenate all batches
    # Shape: (Total_Samples, 107, 5)
    all_preds = np.concatenate(all_preds, axis=0)

    # 5. Formatting for Submission
    # The submission requires a row for every position (id_seqpos)
    # Total rows = Num_Samples * Seq_Len

    num_samples = all_preds.shape[0]
    seq_len = all_preds.shape[1]  # Should be 107
    num_targets = all_preds.shape[2]  # Should be 5

    # Flatten predictions to (Num_Samples * Seq_Len, 5)
    flat_preds = all_preds.reshape(-1, num_targets)

    # Generate id_seqpos column
    # Repeat IDs: [id1, id1... (107 times), id2, id2...]
    ids_repeated = np.repeat(all_ids, seq_len)

    # Tile sequence positions: [0, 1, ... 106, 0, 1, ... 106]
    seq_pos_tiled = np.tile(np.arange(seq_len), num_samples)

    # Create strings "id_seqpos"
    id_seqpos_list = [f"{i}_{j}" for i, j in zip(ids_repeated, seq_pos_tiled)]

    # 6. Create DataFrame
    submission_df = pd.DataFrame(flat_preds, columns=config.target_cols)

    # Insert id_seqpos at the beginning
    submission_df.insert(0, "id_seqpos", id_seqpos_list)

    # Ensure columns are in the correct order as per sample_submission
    # sample_submission columns: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # config.target_cols is usually defined in this order, but we enforce it just in case.
    required_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    submission_df = submission_df[required_cols]

    # 7. Save
    os.makedirs(config.working_dir, exist_ok=True)
    save_path = config.submission_path
    submission_df.to_csv(save_path, index=False)

    print(f"Submission saved to {save_path}")
    print(f"Submission shape: {submission_df.shape}")
