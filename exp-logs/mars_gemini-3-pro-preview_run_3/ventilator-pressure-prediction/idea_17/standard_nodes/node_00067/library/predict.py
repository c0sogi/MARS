import os
import torch
from library.config import Config
from library.dataset import prepare_datasets
from library.model import DCLKNet, predict_and_submit
from library.utils import get_device


def generate_predictions(
    batch_size=Config.BATCH_SIZE, debug=Config.DEBUG, force_recompute=False
):
    """
    Loads the trained model and generates predictions for the test set.

    Args:
        batch_size (int): Batch size for inference. Defaults to Config.BATCH_SIZE.
        debug (bool): Whether to run in debug mode (subset of data). Defaults to Config.DEBUG.
        force_recompute (bool): Whether to force feature re-engineering. Defaults to False.
    """
    # 1. Update Configuration
    # Update the Config class attributes so library functions use the correct values
    Config.BATCH_SIZE = batch_size
    Config.DEBUG = debug

    print(f"Starting prediction pipeline...")
    print(f"Configuration: Batch Size={Config.BATCH_SIZE}, Debug={Config.DEBUG}")

    # 2. Prepare Datasets
    # We invoke the standard dataset preparation pipeline.
    # This ensures feature engineering (and caching) is identical to training.
    # We unpack the tuple to get just the test_loader.
    _, _, test_loader = prepare_datasets(
        batch_size=batch_size, force_recompute=force_recompute
    )

    # 3. Initialize Model
    device = get_device()
    model = DCLKNet().to(device)

    # 4. Load Trained Weights
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Please ensure training has completed successfully before running predictions."
        )

    print(f"Loading model weights from {model_path}...")
    # Load state dict handling map_location for device compatibility
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # 5. Generate and Save Predictions
    # This function handles the inference loop, ID alignment, and CSV saving
    predict_and_submit(model, test_loader)

    print("Prediction pipeline completed successfully.")
