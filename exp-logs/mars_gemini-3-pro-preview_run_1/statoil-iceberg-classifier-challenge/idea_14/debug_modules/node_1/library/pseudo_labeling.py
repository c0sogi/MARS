import torch
from library.config import Config
from library.dataset import get_test_loader, get_pseudo_label_loader
from library.inference import predict_ensemble
from library.utils import log_message


def generate_pseudo_labels(models, device=Config.DEVICE):
    """
    Generates predictions for the test set using the provided models.
    Uses Test Time Augmentation (TTA) via the inference engine.

    Args:
        models (list or nn.Module): A single model or a list of models (ensemble).
        device (str): Device to run inference on.

    Returns:
        dict: A dictionary mapping image IDs to predicted probabilities (floats).
    """
    # Ensure models is a list to be compatible with predict_ensemble
    if not isinstance(models, list):
        models = [models]

    log_message(f"Generating pseudo-labels using ensemble of {len(models)} model(s)...")

    # Get test loader (standard, no shuffle)
    # We use cached data if available for speed, consistent with the rest of the pipeline
    test_loader = get_test_loader(load_cached_data=True)

    # Generate predictions
    # predict_ensemble handles TTA internally for each model and averages the results
    predictions = predict_ensemble(models, test_loader, device=device)

    log_message(f"Generated predictions for {len(predictions)} test samples.")
    return predictions


def create_combined_loader(models, device=Config.DEVICE, load_cached_data=True):
    """
    Orchestrates the creation of the semi-supervised DataLoader (Cycle 2).
    1. Generates predictions on the test set using the Cycle 1 models.
    2. Filters these predictions based on confidence thresholds.
    3. Combines high-confidence test samples with the original training data.

    Args:
        models (list or nn.Module): The trained Cycle 1 model(s).
        device (str): Device for inference.
        load_cached_data (bool): Whether to use cached raw data for the underlying images.

    Returns:
        DataLoader: A DataLoader containing the combined dataset (Train + Pseudo-Labeled Test).
    """
    # 1. Generate Predictions
    # We need current model predictions to determine which samples are confident enough
    pseudo_preds = generate_pseudo_labels(models, device)

    # 2. Create Combined Loader
    # The filtering logic (checking thresholds) and array concatenation resides in
    # get_pseudo_label_loader to keep data manipulation logic encapsulated in the dataset module.
    log_message("Creating combined DataLoader with pseudo-labels...")

    # This function will:
    # - Load original train data
    # - Load test data
    # - Filter test data using pseudo_preds and Config thresholds
    # - Concatenate and return a shuffled DataLoader
    combined_loader = get_pseudo_label_loader(
        pseudo_preds, load_cached_data=load_cached_data
    )

    return combined_loader
