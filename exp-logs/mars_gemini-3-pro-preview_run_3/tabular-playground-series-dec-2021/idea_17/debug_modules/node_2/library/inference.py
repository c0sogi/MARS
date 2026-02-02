import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import get_device
from library.data_loader import get_dataloaders
from library.model import ParallelDCNResNet


def predict(model, test_loader, device):
    """
    Generates predictions for the test set using the provided model.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to perform inference on.

    Returns:
        list: A list of predicted class indices.
    """
    model.eval()
    all_preds = []

    print("Starting inference...")
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())

    return all_preds


def save_submission(predictions, test_ids, output_path):
    """
    Formats predictions and saves them to a CSV file.

    Args:
        predictions (list): List of predicted class indices.
        test_ids (np.ndarray): Array of corresponding test IDs.
        output_path (str): Path to save the submission CSV.
    """
    # Map predictions back to original labels using Inverse Map
    final_preds = [Config.INVERSE_LABEL_MAP[p] for p in predictions]

    # Create DataFrame
    df_sub = pd.DataFrame({"Id": test_ids, "Cover_Type": final_preds})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    print(f"Saving submission to {output_path}...")
    df_sub.to_csv(output_path, index=False)
    print("Submission saved.")


def run_inference(
    load_cached_data=True,
    model_path=Config.MODEL_PATH,
    output_path=Config.SUBMISSION_PATH,
    debug=Config.DEBUG,
):
    """
    Orchestrates the inference pipeline: loads data, loads model, predicts, and saves.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
        model_path (str): Path to the trained model weights.
        output_path (str): Path to save the submission file.
        debug (bool): If True, runs on a subset of data (via Config modification).
    """
    # Set debug mode in Config so data_loader respects it
    Config.DEBUG = debug

    device = get_device()

    print("Loading test data...")
    # get_dataloaders returns (train, val, test, ids). We only need test and ids.
    _, _, test_loader, test_ids = get_dataloaders(load_cached_data=load_cached_data)

    # Determine input dimension from the dataset attached to the loader
    # test_loader.dataset is a ForestCoverDataset which has .X attribute
    if hasattr(test_loader.dataset, "X"):
        input_dim = test_loader.dataset.X.shape[1]
    else:
        # Fallback if dataset structure changes, though unlikely given library code
        # Get a single batch to check shape
        sample_batch = next(iter(test_loader))
        input_dim = sample_batch.shape[1]

    print(f"Data Loaded. Input Dimension: {input_dim}")

    # Initialize Model Architecture
    # We must match the architecture used during training
    print("Initializing model architecture...")
    model = ParallelDCNResNet(
        input_dim=input_dim,
        num_classes=Config.NUM_CLASSES,
        dcn_rank=Config.DCN_RANK,
        resnet_hidden=Config.RESNET_HIDDEN_DIM,
        resnet_blocks=Config.RESNET_NUM_BLOCKS,
        dropout=Config.DROPOUT_RATE,
    ).to(device)

    # Load Weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please train the model first."
        )

    print(f"Loading model weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Run Prediction
    preds = predict(model, test_loader, device)

    # Save Submission
    save_submission(preds, test_ids, output_path)
