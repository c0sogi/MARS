import torch
import os
from library.config import MODEL_SAVE_PATH, SUBMISSION_SAVE_PATH
from library.utils import get_device
from library.data_loader import get_dataloaders
from library.model import ManufacturingMLP, generate_submission


def predict_and_submit(
    load_cached_data=True, model_path=MODEL_SAVE_PATH, output_path=SUBMISSION_SAVE_PATH
):
    """
    Loads the trained model, generates predictions for the test set,
    and saves the submission file.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
                                 Passed to get_dataloaders.
        model_path (str): Path to the saved model checkpoint.
        output_path (str): Path to save the submission CSV.
    """
    # 1. Setup Device
    device = get_device()

    # 2. Load Data
    # We invoke get_dataloaders to ensure data is preprocessed and loaded correctly.
    # We only need the test_loader for inference.
    print(f"Loading data (Cached={load_cached_data})...")
    _, _, test_loader = get_dataloaders(load_cached_data=load_cached_data)

    # 3. Initialize Model
    # The architecture parameters are pulled from config defaults within the class __init__
    print("Initializing model architecture...")
    model = ManufacturingMLP()
    model.to(device)

    # 4. Load Model Weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    print(f"Loading model weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Set model to evaluation mode
    model.eval()

    # 5. Generate Submission
    # The generate_submission function in library.model handles:
    # - Iterating over the test loader
    # - Applying Sigmoid
    # - Aligning with Test Metadata IDs
    # - Saving to CSV
    generate_submission(model, test_loader, device, output_path=output_path)
