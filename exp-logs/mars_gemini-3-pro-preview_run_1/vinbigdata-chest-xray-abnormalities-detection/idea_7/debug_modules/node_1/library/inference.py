import os
import torch
from torch.utils.data import DataLoader, Subset
import pandas as pd

from library.config import Config
from library.utils import seed_everything
from library.dataset import ThoracicDataset
from library.model import EfficientNetBiFPN
from library.engine import generate_submission


class SubsetWrapper(Subset):
    """
    A wrapper around torch.utils.data.Subset that exposes the 'image_ids'
    attribute of the underlying dataset, sliced to the subset indices.
    This is required because the engine expects to access dataset.image_ids.
    """

    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)

    @property
    def image_ids(self):
        # Access the underlying dataset's image_ids and slice them
        return self.dataset.image_ids[self.indices]


def predict_and_format(model, dataloader, device, output_path):
    """
    Orchestrates the prediction generation and formatting process.

    Implements Gated Inference and Coupled Rescaling by leveraging the
    engine's logic.

    Args:
        model: The loaded PyTorch model.
        dataloader: DataLoader containing test images.
        device: The device (CPU/GPU) to run on.
        output_path: File path to save the submission CSV.
    """
    # The generate_submission function in the engine handles:
    # 1. Model inference
    # 2. Gated Inference (Global Head check)
    # 3. Coupled Rescaling (Box mapping to original dims)
    # 4. Formatting to CSV string
    generate_submission(model, dataloader, device, output_path)


def run_inference(
    checkpoint_path=None,
    subset_size=None,
    batch_size=Config.BATCH_SIZE,
    device=Config.DEVICE,
):
    """
    Main entry point for running inference.

    Args:
        checkpoint_path (str, optional): Path to model weights. Defaults to best_model.pth.
        subset_size (int, optional): Number of samples to use (for debugging).
        batch_size (int): Batch size for the dataloader.
        device (str): Device string (e.g., 'cuda', 'cpu').
    """
    # 1. Setup Environment
    seed_everything(Config.SEED)
    Config.setup()

    print(f"Initializing Inference on {device}...")

    # 2. Load Model
    model = EfficientNetBiFPN()
    model.to(device)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"WARNING: Checkpoint {checkpoint_path} not found. Using random initialization."
        )

    # 3. Prepare Data
    # load_cached_data=True ensures we use the cache mechanism defined in dataset.py
    test_dataset = ThoracicDataset(split="test", load_cached_data=True)

    # Handle subsetting for debugging
    if subset_size is not None and subset_size < len(test_dataset):
        print(f"Subsetting test set to {subset_size} samples.")
        indices = list(range(subset_size))
        test_dataset = SubsetWrapper(test_dataset, indices)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # 4. Run Prediction and Format Output
    output_csv_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    print("Starting prediction loop...")
    predict_and_format(model, test_loader, device, output_csv_path)
    print("Inference completed successfully.")
