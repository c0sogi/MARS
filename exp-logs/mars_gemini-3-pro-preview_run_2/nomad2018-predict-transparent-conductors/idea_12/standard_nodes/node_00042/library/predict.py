import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader

from library.config import (
    CACHE_DIR,
    CHECKPOINT_DIR,
    SUBMISSION_DIR,
    TEST_CSV,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE,
    SEED,
)
from library.utils import CompositionScaler, LogStandardScaler
import library.data
from library.data import process_data, CrystalDataset
from library.model import CrystalGraphConvNet
from library.train import generate_submission


def set_seed(seed):
    """
    Sets random seeds for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_predictions(load_cached_data=True, debug_sample_size=None):
    """
    Generates predictions for the test set using the best trained model.

    Args:
        load_cached_data (bool): Whether to load pre-processed graph data from cache.
        debug_sample_size (int, optional): If set, limits the number of test samples for debugging.
    """
    set_seed(SEED)

    # Apply debug limit to the data processing module if requested
    if debug_sample_size is not None:
        library.data.DEBUG_SAMPLE_SIZE = debug_sample_size
        print(f"Debug mode enabled: limiting test set to {debug_sample_size} samples.")

    # 1. Load Fitted Scalers
    # These must exist from the training phase
    print("Loading scalers from cache...")
    comp_scaler_path = os.path.join(CACHE_DIR, "global_scaler.npz")
    target_scaler_path = os.path.join(CACHE_DIR, "target_scaler.npz")

    if not os.path.exists(comp_scaler_path) or not os.path.exists(target_scaler_path):
        raise FileNotFoundError(
            f"Scalers not found in {CACHE_DIR}. Please run training first to generate them."
        )

    # We load comp_scaler just to pass it to generate_submission, though it won't be used
    comp_scaler = CompositionScaler()
    comp_scaler.load(comp_scaler_path)

    target_scaler = LogStandardScaler()
    target_scaler.load(target_scaler_path)

    # 2. Prepare Test Data
    # We use the shared process_data function which handles caching internally
    print("Processing/Loading test data...")
    test_cache_path = os.path.join(CACHE_DIR, "test_graphs.npz")

    # process_data returns a list of Data objects
    test_data_list = process_data(
        metadata_path=TEST_CSV,
        cache_path=test_cache_path,
        load_cached_data=load_cached_data,
        is_test=True,
    )

    test_dataset = CrystalDataset(test_data_list)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Load Model
    print("Loading best model checkpoint...")
    model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    model = CrystalGraphConvNet().to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    # 4. Generate and Save Submission
    # The generate_submission function in library.train handles the inference loop,
    # inverse scaling, and CSV writing.
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    generate_submission(
        model=model,
        loader=test_loader,
        device=DEVICE,
        comp_scaler=comp_scaler,
        target_scaler=target_scaler,
        output_path=submission_path,
    )

    print("Prediction process completed successfully.")
