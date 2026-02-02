import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, Subset

from library.config import Config
from library.dataset import prepare_datasets
from library.model import WPABiLSTM


def predict(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug_limit=None
):
    """
    Runs inference on the test set using the best saved model and generates a submission file.

    Args:
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of worker processes for data loading.
        debug_limit (int, optional): If set, limits the number of breath sequences to process (for debugging).
    """
    # 1. Setup Configuration and Device
    Config.setup()
    device = torch.device(Config.DEVICE)
    print(f"Inference running on device: {device}")

    # 2. Load Data
    # prepare_datasets handles caching and preprocessing (scaling, feature engineering)
    # We only need the test_dataset
    print("Loading datasets...")
    _, _, test_dataset = prepare_datasets(load_cached_data=True)

    # Optional: Subset for debugging
    if debug_limit is not None:
        print(f"Debug mode: Limiting test dataset to {debug_limit} samples.")
        limit = min(len(test_dataset), debug_limit)
        indices = list(range(limit))
        test_dataset = Subset(test_dataset, indices)

    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 3. Initialize Model and Load Weights
    print("Initializing model...")
    model = WPABiLSTM().to(device)

    model_path = Config.BEST_MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    print(f"Loading model weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Run Inference
    print("Starting inference...")
    all_preds = []

    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device)

            # Forward pass
            # Output shape: (batch, seq_len)
            preds = model(inputs)

            # Flatten to (batch * seq_len) for submission format
            preds_flat = preds.cpu().numpy().flatten()
            all_preds.append(preds_flat)

    # Concatenate all batches
    final_preds = np.concatenate(all_preds)
    print(f"Generated {len(final_preds)} predictions.")

    # 5. Generate Submission File
    print(f"Loading test metadata from {Config.TEST_META}...")
    test_meta = pd.read_csv(Config.TEST_META)

    # ALIGNMENT CRITICAL STEP:
    # The dataset preprocessing in library/dataset.py sorts data by ['breath_id', 'time_step'].
    # The test_dataset therefore yields sequences in this sorted order.
    # We must ensure the metadata aligns with this order.
    # We assume 'id' is monotonic with 'time_step' within a breath.
    test_meta = test_meta.sort_values(["breath_id", "id"]).reset_index(drop=True)

    # Handle debug limit in metadata if necessary
    if debug_limit is not None:
        # The test_dataset was subsetted by breaths (indices of dataset are breaths)
        # Each breath has 80 steps (SEQ_LEN defined in dataset.py).
        # So we need to slice the metadata rows corresponding to those breaths.
        seq_len = 80
        total_rows = debug_limit * seq_len
        test_meta = test_meta.iloc[:total_rows].copy()

    # Validation
    if len(final_preds) != len(test_meta):
        print(
            f"Warning: Prediction count ({len(final_preds)}) does not match metadata row count ({len(test_meta)})."
        )
        # If lengths don't match (e.g. due to jagged last batch or data issues), trim to minimum
        min_len = min(len(final_preds), len(test_meta))
        final_preds = final_preds[:min_len]
        test_meta = test_meta.iloc[:min_len]

    # Create DataFrame
    submission = pd.DataFrame({"id": test_meta["id"], "pressure": final_preds})

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission generation complete.")
