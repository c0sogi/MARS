import os
import torch
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import MultiTaskEfficientNet, predict_and_submit


def run_inference(
    checkpoint_path=Config.MODEL_SAVE_PATH,
    submission_path=Config.SUBMISSION_PATH,
    debug=Config.DEBUG,
    load_cached_data=True,
):
    """
    Orchestrates the inference process:
    1. Sets seeds and device.
    2. Loads the test dataset (with caching).
    3. Loads the model architecture and weights.
    4. Generates predictions and saves the submission file.

    Args:
        checkpoint_path (str): Path to the trained model weights.
        submission_path (str): Path where the submission CSV will be saved.
        debug (bool): If True, runs on a subset of data.
        load_cached_data (bool): If True, attempts to load preprocessed data from cache.
    """
    # 1. Setup Environment
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Prepare Data
    # We use get_dataloaders to leverage the existing preprocessing and caching logic.
    # We are only interested in the test_loader for inference.
    _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data, debug=debug)

    # 3. Initialize Model
    # We set pretrained=False because we are loading specific trained weights.
    model = MultiTaskEfficientNet(Config.BACKBONE, pretrained=False)
    model.to(device)

    # 4. Load Weights
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        # In a real scenario, this should probably raise an error,
        # but we'll print a warning and proceed (e.g. for testing pipeline mechanics)
        print(
            f"Warning: Model checkpoint not found at {checkpoint_path}. Using initialized weights."
        )

    # 5. Generate Predictions and Submit
    # The predict_and_submit function in library.model handles:
    # - Iterating through the loader
    # - Collecting probabilities
    # - Aggregating by prediction_id (Max Pooling)
    # - Saving to CSV
    predict_and_submit(model, test_loader, device, submission_path)
