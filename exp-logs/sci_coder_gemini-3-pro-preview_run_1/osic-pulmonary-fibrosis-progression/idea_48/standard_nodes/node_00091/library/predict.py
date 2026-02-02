import os
import torch
import pandas as pd
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import SLHDAN
from library.train import generate_submission


def run_inference(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Executes the inference pipeline for the SLH-DAN model.

    Steps:
    1. Sets random seeds and device.
    2. Loads the test data loader.
    3. Initializes the model and loads the best checkpoint.
    4. Generates predictions using the parametric trajectory logic.
    5. Saves the submission file.

    Args:
        batch_size (int): Batch size for inference.
        num_workers (int): Number of worker threads for data loading.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Initializing Inference on {device}...")

    # 2. Data Loading
    # get_dataloaders returns (train, val, test). We only need the test loader.
    # The loader handles DICOM caching and metadata preparation automatically.
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(batch_size=batch_size, num_workers=num_workers)

    # 3. Model Initialization
    print("Initializing model architecture...")
    model = SLHDAN().to(device)

    # Load Weights
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.BEST_MODEL_PATH}. "
            "Please ensure training has completed successfully."
        )

    print(f"Loading checkpoint from {Config.BEST_MODEL_PATH}...")
    checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)

    # 4. Generate Predictions
    # generate_submission handles the forward pass with temporal anchors
    # (Base FVC, Base Week) to compute the specific FVC/Confidence for the target week.
    print("Generating submission predictions...")
    sub_df = generate_submission(model, test_loader, device)

    # 5. Save Output
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission successfully saved to {Config.SUBMISSION_PATH}")
    print("-" * 30)
    print(sub_df.head())
    print("-" * 30)
