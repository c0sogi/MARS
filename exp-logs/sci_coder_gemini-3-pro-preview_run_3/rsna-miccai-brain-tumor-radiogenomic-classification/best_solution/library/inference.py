import os
import torch
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders
from library.model import BraTS25DEfficientNet
from library.train import predict_and_submit


def run_inference(
    checkpoint_path: str = "./working/idea_3/best_model.pth",
    output_path: str = "./submission/submission.csv",
    batch_size: int = 8,
    load_cached_data: bool = True,
):
    """
    Runs inference on the test set using the Multi-View Ensemble strategy and generates a submission file.

    Args:
        checkpoint_path (str): Path to the trained model weights (default: ./working/idea_3/best_model.pth).
        output_path (str): Destination path for the submission CSV (default: ./submission/submission.csv).
        batch_size (int): Batch size for the test data loader.
        load_cached_data (bool): If True, attempts to load preprocessed data from cache; otherwise processes from scratch.
    """
    # 1. Setup Environment
    seed_everything(42)
    device = get_device()

    # 2. Load Data
    # get_dataloaders handles the caching logic internally.
    # We ignore train/val loaders as we are in inference mode.
    _, _, test_loader, test_ids = get_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data
    )

    # 3. Initialize Model
    # We use pretrained=False because we are about to load our own trained weights.
    # This avoids potential connection errors or overhead from downloading ImageNet weights.
    model = BraTS25DEfficientNet(pretrained=False)
    model.to(device)

    # 4. Load Weights
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    # 5. Run Prediction
    # predict_and_submit implements the Multi-View Ensemble (Even + Odd slices)
    # and handles saving the CSV to the specified output_path.
    predict_and_submit(model, test_loader, test_ids, device, output_path)
