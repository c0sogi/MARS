import os
import torch
from library.config import Config
from library.dataset import get_dataloaders
from library.model import BiGRUModel, generate_submission


def predict_and_submit(load_cached_data=True, batch_size=None):
    """
    Orchestrates the inference pipeline: loads data, loads the trained model,
    generates predictions, and saves the submission file.

    Args:
        load_cached_data (bool): If True, attempts to load preprocessed test data
                                 from cache. If False, reprocesses raw data.
        batch_size (int, optional): Batch size for the test loader. Defaults to Config.BATCH_SIZE.
    """
    print("==================================================")
    print("INFERENCE PIPELINE")
    print("==================================================")

    # 1. Load Test Data
    # get_dataloaders returns (train_loader, val_loader, test_loader, test_meta)
    # We only need the test components for inference.
    print(f"Loading test data (Cached: {load_cached_data})...")
    _, _, test_loader, test_meta = get_dataloaders(
        load_cached_data=load_cached_data, batch_size=batch_size
    )

    print(f"Test data loaded. Samples: {len(test_loader.dataset)}")

    # 2. Initialize Model
    print("Initializing model...")
    device = torch.device(Config.DEVICE)
    model = BiGRUModel().to(device)

    # 3. Load Model Weights
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model weights not found at {Config.MODEL_PATH}. "
            "Please run the training pipeline first."
        )

    print(f"Loading model weights from {Config.MODEL_PATH}...")
    state_dict = torch.load(Config.MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)

    # 4. Generate Submission
    # This function (from library.model) handles:
    # - Prediction loop
    # - Retrieval of WLS baselines
    # - Reconstruction of absolute Lat/Lon from predicted metric residuals
    # - Saving to Config.SUBMISSION_PATH
    print("Starting prediction and submission generation...")
    generate_submission(model, test_loader, test_meta)

    print("Inference pipeline completed successfully.")
