import torch
import os
from library.config import Config
from library.data import get_loaders
from library.model import CGCNN
from library.train import generate_submission


def generate_predictions(load_cached_data=True):
    """
    Loads the trained model and test data, generates predictions, and saves them to a CSV file.

    Args:
        load_cached_data (bool): Whether to load pre-processed graph data from cache.
                                 If False or cache missing, data will be processed from scratch.
    """
    # Hardware setup
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Ensure checkpoint exists
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.BEST_MODEL_PATH}. Please train the model first."
        )

    # 1. Data Loading
    # We obtain the test_loader for iterating over test samples.
    # We also need the scaler, which is fitted on the training data within get_loaders,
    # to inverse-transform the model's standardized predictions back to eV.
    print("Initializing DataLoaders and Scaler...")
    # get_loaders returns: train_loader, val_loader, test_loader, scaler
    _, _, test_loader, scaler = get_loaders(load_cached_data=load_cached_data)

    # 2. Model Initialization
    print("Initializing Model...")
    model = CGCNN(config=Config).to(device)

    # 3. Load Model Weights
    print(f"Loading model weights from {Config.BEST_MODEL_PATH}...")
    checkpoint = torch.load(
        Config.BEST_MODEL_PATH, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint)

    # 4. Generate and Save Submission
    # The generate_submission function handles the inference loop, inverse transformation,
    # and formatting the output into the required CSV format.
    generate_submission(
        model=model,
        loader=test_loader,
        device=device,
        scaler=scaler,
        output_path=Config.SUBMISSION_PATH,
    )

    print("Prediction process completed.")
