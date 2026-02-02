import os
import torch
import numpy as np
import random
from torch.utils.data import DataLoader

from library.config import Config
from library.model import UnifiedSegFormer
from library.data import InkDataset
from library.engine import generate_submission


def set_seed(seed):
    """
    Sets the random seed for reproducibility across numpy, random, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_inference(
    checkpoint_path=Config.BEST_MODEL_PATH,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    limit_size=None,
):
    """
    Orchestrates the inference pipeline for the Vesuvius Challenge.

    This function:
    1. Initializes the UnifiedSegFormer model.
    2. Loads trained weights.
    3. Prepares the test data loader which yields multi-view Z-stacks.
    4. Calls the engine to generate predictions using Max-Fusion and save the submission.

    Args:
        checkpoint_path (str): Path to the saved model weights.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.
        limit_size (int, optional): If provided, limits the dataset size (useful for debugging).
    """
    # 1. Setup Environment
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Starting Inference on device: {device}")

    # 2. Initialize Model
    print("Initializing UnifiedSegFormer model...")
    model = UnifiedSegFormer()
    model.to(device)

    # 3. Load Weights
    if os.path.exists(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"WARNING: Checkpoint file not found at {checkpoint_path}.")
        print(
            "Inference will proceed with random weights (results will be meaningless)."
        )

    # 4. Prepare Data
    # The InkDataset in 'test' mode returns a stack of views corresponding to
    # Config.INFERENCE_Z_STARTS (e.g., 16, 20, 24).
    print("Preparing test dataset...")
    test_dataset = InkDataset(mode="test", load_cached_data=True, limit_size=limit_size)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 5. Generate Submission
    # The generate_submission function in library.engine handles:
    # - Iterating over the loader
    # - Running the model on all Z-views
    # - Max-Fusion (aggregating probabilities across views)
    # - Reconstructing full fragment masks
    # - RLE encoding and saving to CSV
    generate_submission(model, test_loader, device)

    print("Inference completed successfully.")
