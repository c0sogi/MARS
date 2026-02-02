import os
import torch
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import SpatialSymmetryDifferenceNet
from library.train import generate_submission


def predict(
    weights_path=Config.MODEL_SAVE_PATH,
    load_cached_data=True,
    debug=False,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Executes the inference pipeline: loads data, initializes the model,
    generates predictions, aggregates them by prediction_id, and saves the submission.

    Args:
        weights_path (str): Path to the saved model weights (.pth file).
                            Defaults to the path defined in Config.
        load_cached_data (bool): If True, attempts to load processed metadata from cache.
                                 If False, re-processes the metadata.
        debug (bool): If True, runs inference on a small subset of the test data for debugging.
        debug_sample_size (int): The number of samples to use when debug is True.
    """
    # 1. Ensure Reproducibility
    set_seed(Config.SEED)

    # 2. Configure Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}")

    # 3. Prepare Data
    # get_dataloaders returns (train, val, test). We only need the test loader.
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(
        load_cached_data=load_cached_data,
        debug=debug,
        debug_sample_size=debug_sample_size,
    )

    # 4. Initialize Model
    print("Initializing Spatial Symmetry-Difference Network...")
    model = SpatialSymmetryDifferenceNet()
    model.to(device)

    # 5. Load Weights
    if os.path.exists(weights_path):
        print(f"Loading model weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Weights file not found at {weights_path}.")
        print("Using randomly initialized weights (Expect random predictions).")

    # 6. Generate and Save Submission
    # This function handles the forward pass, sigmoid activation,
    # aggregation (Max per prediction_id), and CSV saving.
    print(f"Generating predictions for {len(test_loader.dataset)} samples...")
    generate_submission(
        model=model,
        loader=test_loader,
        device=device,
        output_path=Config.SUBMISSION_PATH,
    )

    print("Inference pipeline completed successfully.")
