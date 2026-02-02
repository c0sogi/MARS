import os
import torch
from library.config import Config
from library.model import CactusResNet
from library.dataset import get_dataloaders
from library.utils import save_submission, load_checkpoint
from library.train import inference_tta


def run_prediction(checkpoint_path=None, debug=Config.DEBUG):
    """
    Loads the trained model and generates predictions on the test set using
    Test-Time Augmentation (TTA).

    Args:
        checkpoint_path (str, optional): Path to the saved model checkpoint.
                                         Defaults to the best model in working directory.
        debug (bool): If True, runs inference on a small subset of the test data.
    """
    # Determine device
    device = torch.device(Config.DEVICE)

    # Default checkpoint path if not provided
    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file not found at {checkpoint_path}. Please run training first."
        )

    print(f"Initializing inference on {device}...")

    # 1. Load Test Data
    # We retrieve the dictionary of loaders and select 'test'
    dataloaders = get_dataloaders(debug=debug)
    test_loader = dataloaders["test"]

    # 2. Initialize Model
    # Must match the architecture used during training
    model = CactusResNet(num_classes=Config.NUM_CLASSES).to(device)

    # 3. Load Weights
    print(f"Loading model weights from {checkpoint_path}...")
    model = load_checkpoint(model, checkpoint_path, device)

    # 4. Run Inference with TTA
    # inference_tta handles the averaging of original + flipped predictions
    print("Starting Test-Time Augmentation (TTA) inference...")
    predictions = inference_tta(model, test_loader, device)

    # 5. Prepare Submission Data
    # Retrieve the list of file IDs corresponding to the predictions
    test_ids = test_loader.dataset.df["id"].values

    # 6. Save Submission
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    save_submission(test_ids, predictions, Config.SUBMISSION_PATH)

    print("Prediction process completed successfully.")
