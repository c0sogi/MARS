import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.config import Config
from library.dataset import load_data
from library.model import WideResBiGRU


def set_seed(seed):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def predict(device=None, subset_size=None):
    """
    Runs inference on the test set and generates the submission file.

    Args:
        device (str, optional): Device to run inference on ('cpu' or 'cuda').
                                Defaults to Config.DEVICE.
        subset_size (int, optional): If provided, limits the test set size for debugging.
    """
    # 1. Setup
    if device is None:
        device = Config.DEVICE

    set_seed(Config.SEED)
    print(f"Running inference on device: {device}")

    # 2. Load Data
    # load_data handles caching and processing from metadata/test.parquet
    print("Loading test dataset...")
    test_dataset = load_data(
        mode="test", load_cached_data=True, subset_size=subset_size
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Initialize Model
    print("Initializing model...")
    model = WideResBiGRU().to(device)

    # Load best weights
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.BEST_MODEL_PATH}"
        )

    state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Inference Loop
    print("Generating predictions...")
    all_ids = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(
            test_loader, disable=None
        ):  # disable=None lets tqdm decide based on output stream
            seq = batch["sequence"].to(device)
            loop = batch["loop_type"].to(device)
            dist = batch["distance"].to(device)
            ids = batch["id"]  # List of IDs

            # Forward pass -> (Batch, Seq_Len, 3)
            # The model outputs predictions for the full sequence length (107)
            preds = model(seq, loop, dist)

            all_preds.append(preds.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate all predictions: (N_Samples, 107, 3)
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
    else:
        all_preds = np.empty((0, Config.SEQ_LEN, 3))

    # 5. Format Submission
    print("Formatting submission...")
    submission_rows = []

    # The columns predicted by the model
    pred_cols = Config.TARGET_COLS  # ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']

    # The full set of columns required for submission
    req_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Map predicted columns to their index in the model output (0, 1, 2)
    col_indices = {col: i for i, col in enumerate(pred_cols)}

    # Iterate through samples and positions to build the long-format dataframe
    # Using a list of dictionaries is generally memory intensive but clear.
    # For 240 samples * 107 positions = ~25k rows, this is fast enough.

    # Pre-calculate column indices to avoid dict lookup in inner loop
    req_col_map = []
    for col in req_cols:
        if col in col_indices:
            req_col_map.append(
                (col, col_indices[col], True)
            )  # (Name, Index, IsPredicted)
        else:
            req_col_map.append((col, -1, False))

    for i, sample_id in enumerate(all_ids):
        sample_preds = all_preds[i]  # Shape: (107, 3)

        for pos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{pos}"

            # Construct row data
            row_data = {"id_seqpos": row_id}

            for col_name, col_idx, is_predicted in req_col_map:
                if is_predicted:
                    row_data[col_name] = float(sample_preds[pos, col_idx])
                else:
                    row_data[col_name] = 0.0

            submission_rows.append(row_data)

    # Create DataFrame
    df_sub = pd.DataFrame(submission_rows)

    # 6. Save
    # Ensure directory exists (Config.SUBMISSION_FILE includes the directory)
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
    print(f"Submission shape: {df_sub.shape}")
