import torch
import os
from library.config import (
    MODEL_PATH,
    SUBMISSION_FILE,
    BATCH_SIZE,
    THRESHOLD,
    DEVICE,
    set_seed,
)
from library.dataset import get_dataloaders
from library.model import SparseMLP
from library.trainer import generate_submission as trainer_generate_submission


def generate_predictions(
    model_path=MODEL_PATH,
    output_file=SUBMISSION_FILE,
    batch_size=BATCH_SIZE,
    threshold=THRESHOLD,
    device=DEVICE,
    load_cached_data=True,
):
    """
    Manages the prediction pipeline for the test set.

    Steps:
    1. Loads the test data loader and feature engineer (using cached artifacts).
    2. Initializes the SparseMLP model architecture.
    3. Loads the trained model weights.
    4. Generates predictions and saves them to the submission CSV.

    Args:
        model_path (str): Path to the saved model weights (.pth file).
        output_file (str): Path where the submission CSV will be saved.
        batch_size (int): Batch size for the DataLoader.
        threshold (float): Probability threshold for converting logits to binary tags.
        device (str): Device to run inference on ('cuda' or 'cpu').
        load_cached_data (bool): Whether to load preprocessed data from disk cache.
    """
    set_seed()

    print(f"Starting inference pipeline on device: {device}")

    # 1. Load Data and Feature Engineer
    # We use get_dataloaders to ensure consistent preprocessing with the training phase.
    # We discard train/val loaders as we only need the test loader and the feature engineer.
    print("Loading test data and feature engineer...")
    _, _, test_loader, feature_engineer = get_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data
    )

    # 2. Initialize Model
    print("Initializing model architecture...")
    model = SparseMLP()
    model.to(device)

    # 3. Load Model Weights
    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Please ensure the model is trained and saved before running inference."
        )

    # 4. Generate Predictions
    # This function handles the inference loop, thresholding, decoding, and file saving.
    print(f"Generating predictions with threshold {threshold}...")
    trainer_generate_submission(
        model=model,
        test_loader=test_loader,
        device=device,
        feature_engineer=feature_engineer,
        threshold=threshold,
        submission_file=output_file,
    )

    print(f"Inference complete. Submission saved to {output_file}")
