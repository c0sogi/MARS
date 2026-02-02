import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed
from library.dataset import CachedDataset, collate_fn
from library.model import DCAN
from library.engine import Engine


def generate_submission(batch_size=None, num_workers=None):
    """
    Loads the trained model and test dataset, generates predictions,
    and saves the submission file.

    Args:
        batch_size (int, optional): Batch size for inference. Defaults to Config.BATCH_SIZE.
        num_workers (int, optional): Number of workers for DataLoader. Defaults to Config.NUM_WORKERS.

    Returns:
        str: Path to the generated submission file.
    """
    # 1. Setup Configuration
    config = Config()
    set_seed(config.SEED)
    device = torch.device(config.DEVICE)

    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if num_workers is None:
        num_workers = config.NUM_WORKERS

    print(f"Starting inference on device: {device}")

    # 2. Load Test Dataset
    # CachedDataset handles loading from parquet or processing from scratch if needed
    print("Loading test dataset...")
    test_dataset = CachedDataset(mode="test", load_cached_data=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == "cuda" else False,
    )
    print(f"Test dataset loaded: {len(test_dataset)} samples.")

    # 3. Initialize Model
    print("Initializing model architecture...")
    model = DCAN()
    model.to(device)

    # 4. Load Trained Weights
    if os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Loading model weights from {config.MODEL_SAVE_PATH}...")
        state_dict = torch.load(config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(
            f"Model checkpoint not found at {config.MODEL_SAVE_PATH}. Cannot perform inference."
        )

    # 5. Initialize Engine
    engine = Engine(model=model, device=device)

    # 6. Generate Predictions
    # We pass test_dataset.df as raw_df because it contains the 'code_ids' and 'markdown_ids'
    # columns required by the engine to reconstruct the cell order strings.
    print("Running prediction loop...")
    engine.predict(test_loader, raw_df=test_dataset.df)

    print(f"Inference complete. Submission saved to {config.SUBMISSION_PATH}")
    return config.SUBMISSION_PATH
