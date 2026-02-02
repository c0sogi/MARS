import torch
import os
import pandas as pd
from library.config import Config
from library.utils import seed_everything
from library.engine import generate_submission


def predict_and_submit(model, test_loader, label_encoder, device):
    """
    Manages the generation of predictions for the test set.

    This function acts as a high-level orchestrator for the inference process.
    It ensures reproducibility and calls the engine's submission generation logic.

    Args:
        model (torch.nn.Module): The trained model (e.g., HotelClassifier).
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        label_encoder (np.ndarray or list): An array-like structure where the index
                                            corresponds to the integer class label and
                                            the value is the original hotel_id string.
        device (torch.device): The computation device (CPU or CUDA).
    """
    # Set random seeds for reproducibility during inference
    seed_everything(Config.SEED)

    # Use the provided engine function to generate predictions and save the submission file.
    # The 'label_encoder' argument serves as the 'classes' mapping required by generate_submission.
    generate_submission(
        model=model, test_loader=test_loader, classes=label_encoder, device=device
    )
