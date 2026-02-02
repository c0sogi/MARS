import os
import torch
from torch.utils.data import DataLoader
from library.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    MODEL_SAVE_PATH,
    SEED,
)
from library.utils import set_seed, collate_fn
from library.dataset import NotebookDataset
from library.model import CAAN
from library.engine import Engine


def predict_and_rank(debug=False):
    """
    Runs the inference pipeline: loads the test dataset and model, generates
    predictions using the soft-ranking strategy, and saves the submission file.

    Args:
        debug (bool): If True, processes a smaller subset of the test data for debugging.
    """
    # 1. Set reproducible state
    set_seed(SEED)

    # 2. Prepare Test Data
    # NotebookDataset handles caching internally.
    # shuffle=False is strictly required for the Engine to map batch indices back to dataset IDs.
    print("Initializing test dataset...")
    test_dataset = NotebookDataset(split="test", load_cached_data=True, debug=debug)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=(DEVICE == "cuda"),
    )

    # 3. Load Model
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model weights not found at {MODEL_SAVE_PATH}. "
            "Please ensure the model has been trained and saved."
        )

    print(f"Loading model weights from {MODEL_SAVE_PATH}...")
    model = CAAN()
    model.to(DEVICE)

    # Load weights
    state_dict = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    # 4. Generate Predictions
    # The Engine handles the forward pass, expected rank calculation,
    # order reconstruction, and writing to submission.csv.
    engine = Engine(model=model, device=DEVICE)
    engine.generate_submission(test_loader, test_dataset)

    print("Inference process completed.")
