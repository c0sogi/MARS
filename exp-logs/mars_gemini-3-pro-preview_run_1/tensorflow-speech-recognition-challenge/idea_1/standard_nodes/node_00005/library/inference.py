import os
import torch
from library.config import Config
from library.model import SpectroCNN
from library.trainer import generate_submission


def predict_submission(weights_path=None, load_cached_data=True):
    """
    Loads the best trained model and generates predictions for the test set.
    The predictions are saved to the submission file path defined in Config.

    Args:
        weights_path (str, optional): Path to the model weights file.
            If None, uses the default best model path from Config.
        load_cached_data (bool): Whether to use cached metadata for the test set.
            Defaults to True.
    """
    # Set device based on Config
    device = Config.DEVICE

    # Resolve weights path
    if weights_path is None:
        weights_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights file not found at: {weights_path}")

    # Initialize the model architecture
    # We use the same architecture parameters as defined in Config
    model = SpectroCNN(num_classes=Config.NUM_CLASSES)
    model.to(device)

    # Load the trained weights
    # map_location ensures weights are loaded to the correct device (e.g. if trained on GPU but running on CPU)
    try:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights from {weights_path}: {e}")

    # Generate submission using the utility from trainer.py
    # This function handles:
    # 1. Initializing the Test Dataset and DataLoader
    # 2. Running the inference loop
    # 3. Mapping numeric predictions to labels
    # 4. Saving the result to submission.csv
    generate_submission(model, device, load_cached_data=load_cached_data)
