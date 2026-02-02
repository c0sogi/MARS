import os
import torch
from library.config import WORKING_DIR, SUBMISSION_DIR, BATCH_SIZE
from library.data import get_loaders
from library.model import BA_ADS_Model
from library.train import generate_submission


def generate_predictions(
    model_path=None,
    output_path=None,
    batch_size=BATCH_SIZE,
    debug_mode=False,
    load_cached_data=True,
    device=None,
):
    """
    Generates predictions for the test set using the trained BA-ADS model.

    Args:
        model_path (str, optional): Path to the saved model weights.
                                    Defaults to 'best_model.pt' in the working directory.
        output_path (str, optional): Path to save the submission CSV.
                                     Defaults to 'submission.csv' in the submission directory.
        batch_size (int): Batch size for inference. Defaults to config BATCH_SIZE.
        debug_mode (bool): If True, runs on a small subset of the data.
        load_cached_data (bool): Whether to attempt loading pre-processed data from cache.
        device (torch.device, optional): Device to run inference on. Auto-detected if None.
    """
    # Set defaults for paths if not provided
    if model_path is None:
        model_path = os.path.join(WORKING_DIR, "best_model.pt")

    if output_path is None:
        output_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running inference on device: {device}")

    # 1. Load Test Data
    # We use get_loaders to ensure consistency with the training pipeline.
    # This handles feature extraction, scaling (using saved scalers), and batching.
    # We ignore the train and validation loaders.
    print("Loading and processing test data...")
    _, _, test_loader = get_loaders(
        batch_size=batch_size, debug_mode=debug_mode, load_cached_data=load_cached_data
    )

    # 2. Initialize Model
    print("Initializing BA-ADS model architecture...")
    model = BA_ADS_Model()
    model.to(device)

    # 3. Load Trained Weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights file not found at: {model_path}. Please train the model first."
        )

    print(f"Loading model weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # 4. Generate and Save Predictions
    # The generate_submission function in library.train handles:
    # - The inference loop
    # - Inverse transformation of targets (expm1)
    # - Formatting and saving the CSV file
    print(f"Generating predictions and saving to {output_path}...")
    generate_submission(model, test_loader, output_path=output_path, device=device)

    print("Prediction pipeline completed successfully.")
