import os
import torch
import numpy as np
import random
from library.config import Config
from library.data import get_dataloaders
from library.model import AsymmetricEfficientNet, generate_submission


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def predict(load_cached_data=True):
    """
    Runs the inference pipeline:
    1. Loads test data (using cache if available).
    2. Loads the trained model architecture and weights.
    3. Generates predictions with TTA.
    4. Saves submission.csv.

    Args:
        load_cached_data (bool): Whether to attempt loading pre-processed data from cache.
    """
    # Set reproducibility
    set_seed(Config.SEED)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # 1. Data Loading
    # We use get_dataloaders to ensure consistent processing logic with training.
    # We only need the test_loader and test_ids.
    print("Loading data...")
    _, _, test_loader, test_ids = get_dataloaders(load_cached_data=load_cached_data)

    # 2. Model Initialization
    print("Initializing model...")
    model = AsymmetricEfficientNet()
    model = model.to(device)

    # 3. Load Weights
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Loading model weights from {Config.BEST_MODEL_PATH}...")
        try:
            state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Proceeding with random weights (Predictions will be invalid).")
    else:
        print(f"Warning: Checkpoint not found at {Config.BEST_MODEL_PATH}.")
        print("Proceeding with random weights (Predictions will be invalid).")

    # 4. Generate Submission
    # This function handles TTA (Original + HFlip + VFlip) and CSV saving
    generate_submission(model, test_loader, test_ids, device)
