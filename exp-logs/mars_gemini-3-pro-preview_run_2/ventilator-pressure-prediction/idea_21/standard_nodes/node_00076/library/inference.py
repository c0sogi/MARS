import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed
from library.dataset import load_data
from library.model import FPBC_BiLSTM


def run_inference(load_cached_data=True, batch_size=Config.BATCH_SIZE):
    """
    Runs inference on the test set using the best trained model and generates the submission file.

    Args:
        load_cached_data (bool): If True, attempts to use cached preprocessed data.
                                 If False, forces data reprocessing.
        batch_size (int): The batch size to use for inference.
    """
    # 1. Environment Setup
    device = torch.device(Config.DEVICE)
    set_seed(Config.SEED)

    print("Initializing inference pipeline...")

    # 2. Data Loading
    # We use the centralized load_data function to ensure feature engineering and scaling
    # are identical to the training phase.
    print("Loading test data...")
    # load_data returns (train, val, test). We only need test.
    _, _, test_dataset = load_data(load_cached_data=load_cached_data)

    # 3. Model Loading
    print(f"Loading model checkpoint from {Config.BEST_MODEL_PATH}...")
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Best model not found at {Config.BEST_MODEL_PATH}. Please train the model first."
        )

    model = FPBC_BiLSTM().to(device)
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # 4. Inference Loop
    print(f"Starting inference with batch size {batch_size}...")
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    all_preds = []

    with torch.no_grad():
        for X, u_out in test_loader:
            X = X.to(device)
            u_out = u_out.to(device)

            # Forward pass
            # Output shape: (Batch, Seq)
            preds = model(X, u_out)

            # Flatten to match the submission format (row-wise)
            # (Batch, Seq) -> (Batch * Seq)
            preds_flat = preds.view(-1).cpu().numpy()
            all_preds.append(preds_flat)

    # Concatenate all batch predictions into a single array
    final_predictions = np.concatenate(all_preds)

    # 5. Submission Generation
    # We need to map the flat predictions back to the 'id' column.
    # The load_data function guarantees that Config.TEST_CACHE_PATH exists and contains
    # the dataframe sorted exactly as the dataset was created (Breath ID, then Time).
    print(f"Loading ID mapping from {Config.TEST_CACHE_PATH}...")
    if not os.path.exists(Config.TEST_CACHE_PATH):
        raise FileNotFoundError(
            f"Test cache file missing at {Config.TEST_CACHE_PATH}. Cannot align IDs."
        )

    df_test = pd.read_parquet(Config.TEST_CACHE_PATH)

    # Integrity Check
    if len(df_test) != len(final_predictions):
        raise ValueError(
            f"Prediction mismatch: Test data has {len(df_test)} rows, "
            f"but model generated {len(final_predictions)} predictions."
        )

    print("Constructing submission dataframe...")
    submission = pd.DataFrame(
        {Config.ID_COL: df_test[Config.ID_COL], Config.TARGET_COL: final_predictions}
    )

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission successfully saved to {Config.SUBMISSION_PATH}")
