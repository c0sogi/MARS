import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.models import DualStreamNetwork
from library.data_loader import process_row, EEGDataset

# Attempt to import joblib for parallel processing
try:
    from joblib import Parallel, delayed

    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


def get_test_data(load_cached_data=True):
    """
    Loads or processes the test data with caching mechanism.

    Args:
        load_cached_data (bool): If True, attempts to load from disk cache first.

    Returns:
        tuple: (eeg_data, spec_data, target_data) as numpy arrays.
    """
    # Define cache paths
    eeg_cache_path = os.path.join(Config.CACHE_DIR, "test_eeg.npy")
    spec_cache_path = os.path.join(Config.CACHE_DIR, "test_spec.npy")
    # Targets are dummy for test set, but required by EEGDataset structure
    target_cache_path = os.path.join(Config.CACHE_DIR, "test_targets.npy")

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Check if cache files exist
    cache_exists = (
        os.path.exists(eeg_cache_path)
        and os.path.exists(spec_cache_path)
        and os.path.exists(target_cache_path)
    )

    if load_cached_data and cache_exists:
        print(f"Loading cached test data from {Config.CACHE_DIR}...")
        # Use mmap_mode='r' to keep memory usage low
        eeg_all = np.load(eeg_cache_path, mmap_mode="r")
        spec_all = np.load(spec_cache_path, mmap_mode="r")
        targets_all = np.load(target_cache_path, mmap_mode="r")
    else:
        print("Processing test data from scratch...")
        # Load metadata
        df = pd.read_csv(Config.TEST_CSV)

        # Handle debug mode
        if Config.DEBUG:
            df = df.head(Config.DEBUG_SUBSET_SIZE)

        rows = [r for _, r in df.iterrows()]

        # Process rows (Parallel or Sequential)
        if HAS_JOBLIB:
            results = Parallel(n_jobs=Config.NUM_WORKERS, backend="loky")(
                delayed(process_row)(row, Config.INPUT_DIR, is_test=True)
                for row in rows
            )
        else:
            results = [process_row(row, Config.INPUT_DIR, is_test=True) for row in rows]

        # Unpack results
        eeg_list, spec_list, target_list = zip(*results)

        # Stack into arrays
        eeg_all = np.stack(eeg_list)
        spec_all = np.stack(spec_list)
        targets_all = np.stack(target_list)

        # Save to cache
        print(f"Saving processed test data to {Config.CACHE_DIR}...")
        np.save(eeg_cache_path, eeg_all)
        np.save(spec_cache_path, spec_all)
        np.save(target_cache_path, targets_all)

    return eeg_all, spec_all, targets_all


def predict_test_set(load_cached_data=True):
    """
    Runs inference on the test set and generates submission.csv.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Inference using device: {device}")

    # 2. Load Data
    test_eeg, test_spec, test_targets = get_test_data(load_cached_data=load_cached_data)

    # Create Dataset and DataLoader
    test_dataset = EEGDataset(test_eeg, test_spec, test_targets, augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # Must be False to align with metadata order
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Load Model
    print(f"Loading model from {Config.MODEL_PATH}...")
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_PATH}. Please train the model first."
        )

    # Initialize model structure (pretrained=False because we load custom weights)
    model = DualStreamNetwork(num_classes=Config.N_CLASSES, pretrained=False)

    # Load weights
    state_dict = torch.load(Config.MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. Inference Loop
    all_probs = []
    print(f"Starting inference on {len(test_dataset)} samples...")

    with torch.no_grad():
        for eeg, spec, _ in test_loader:
            eeg = eeg.to(device)
            spec = spec.to(device)

            # Forward pass
            logits = model(eeg, spec)

            # Apply Softmax to get probabilities (logits -> probs)
            probs = F.softmax(logits, dim=1)

            # Move to CPU and store
            all_probs.append(probs.cpu().numpy())

    # Concatenate all batches
    final_probs = np.concatenate(all_probs, axis=0)

    # 5. Generate Submission File
    # Load test metadata to get eeg_ids
    df_test = pd.read_csv(Config.TEST_CSV)
    if Config.DEBUG:
        df_test = df_test.head(Config.DEBUG_SUBSET_SIZE)

    # Verify alignment
    if len(df_test) != len(final_probs):
        print(
            f"Warning: Mismatch between test metadata rows ({len(df_test)}) and predictions ({len(final_probs)})."
        )
        # Truncate to minimum length to allow saving (though this indicates an upstream issue)
        min_len = min(len(df_test), len(final_probs))
        df_test = df_test.iloc[:min_len]
        final_probs = final_probs[:min_len]

    # Create DataFrame
    submission = pd.DataFrame(final_probs, columns=Config.SUBMISSION_COLS)

    # Insert eeg_id as the first column
    submission.insert(0, "eeg_id", df_test["eeg_id"])

    # Save to CSV
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("First 5 rows of submission:")
    print(submission.head())
