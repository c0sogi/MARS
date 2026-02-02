import os
import torch
from library.config import Config
from library.model import D2N
from library.model import generate_submission as lib_generate_submission
from library.data_loader import get_data_loaders


def generate_submission(
    load_cached_data=True, batch_size=Config.BATCH_SIZE, max_samples=None
):
    """
    Loads the trained model state, iterates through the test DataLoader to compute
    probability scores, and formats the results into a pandas DataFrame.
    It finally saves the predictions to a CSV file in the required submission format.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
        batch_size (int): Batch size for inference.
        max_samples (int, optional): Limit the number of samples for debugging.
    """
    # Ensure model checkpoint exists before proceeding
    if not os.path.exists(Config.MODEL_CHECKPOINT):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_CHECKPOINT}. "
            "Please run the training script first."
        )

    # 1. Load Data
    # We only require the test loader and test IDs for inference.
    # The get_data_loaders function handles caching and preprocessing internally.
    _, _, test_loader, test_ids = get_data_loaders(
        batch_size=batch_size,
        load_cached_data=load_cached_data,
        max_samples=max_samples,
    )

    # 2. Initialize Model Architecture
    # Must match the architecture used during training
    model = D2N(
        input_dim=Config.INPUT_DIM,
        hidden_units=Config.HIDDEN_UNITS,
        dropout_rate=Config.DROPOUT_RATE,
    )

    # 3. Load Trained Weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the state dictionary from the checkpoint file
    state_dict = torch.load(Config.MODEL_CHECKPOINT, map_location=device)
    model.load_state_dict(state_dict)

    # 4. Generate Predictions and Save Submission
    # Delegate to the library function which handles the inference loop,
    # device management, and saving the CSV to Config.SUBMISSION_PATH.
    lib_generate_submission(model, test_loader, test_ids)
