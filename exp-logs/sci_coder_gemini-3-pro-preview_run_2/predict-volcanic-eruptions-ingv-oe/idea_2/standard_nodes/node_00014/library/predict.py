import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, TargetScaler
from library.dataset import get_dataloaders
from library.model import HybridModel


def generate_predictions(load_cached_data=True, device=None):
    """
    Generates predictions for the test set using the trained HybridModel.

    Args:
        load_cached_data (bool): Whether to use cached features and scalers.
        device (torch.device, optional): Device to run inference on.
    """
    # 1. Setup
    seed_everything(Config.SEED)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # 2. Data Loading
    # get_dataloaders handles feature engineering, caching, and scaler initialization
    loaders = get_dataloaders(load_cached_data=load_cached_data)
    test_loader = loaders["test"]

    # Determine input dimension for MLP branch from the dataset
    # The dataset object has a list of feature columns used
    num_tabular_features = len(test_loader.dataset.feature_cols)
    print(f"Detected {num_tabular_features} tabular features for inference.")

    # 3. Load Scaler
    # We need the scaler fitted during training to inverse transform predictions
    target_scaler = TargetScaler()
    if os.path.exists(Config.TARGET_MEAN_PATH) and os.path.exists(
        Config.TARGET_STD_PATH
    ):
        target_scaler.load(Config.TARGET_MEAN_PATH, Config.TARGET_STD_PATH)
    else:
        raise FileNotFoundError(
            f"Target scaler files not found. Expected at {Config.TARGET_MEAN_PATH} and {Config.TARGET_STD_PATH}. "
            "Ensure the training pipeline has been run successfully."
        )

    # 4. Model Initialization & Loading
    model = HybridModel(num_tabular_features=num_tabular_features)

    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_PATH}. "
            "Ensure the training pipeline has been run successfully."
        )

    print(f"Loading model weights from {Config.MODEL_PATH}...")
    state_dict = torch.load(Config.MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 5. Inference Loop
    print("Starting inference on test set...")
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            # Test dataset returns (spectrogram, tabular_features)
            spec, tab = batch

            spec = spec.to(device)
            tab = tab.to(device)

            # Forward pass
            # Output shape: (Batch_Size,)
            preds_scaled = model(spec, tab)

            # Inverse Transform
            # Convert to numpy and apply inverse scaling
            # TargetScaler handles both tensors and numpy arrays
            preds_original = target_scaler.inverse_transform(preds_scaled)

            # If result is a tensor (from utils implementation), convert to list
            if hasattr(preds_original, "tolist"):
                predictions.extend(preds_original.tolist())
            else:
                predictions.extend(preds_original)

    # 6. Submission Generation
    # Retrieve segment_ids from the dataset metadata
    # The dataset merges metadata with features, preserving order
    test_meta = test_loader.dataset.data
    segment_ids = test_meta["segment_id"].values

    if len(segment_ids) != len(predictions):
        raise ValueError(
            f"Mismatch between number of segment IDs ({len(segment_ids)}) and predictions ({len(predictions)})."
        )

    # Create DataFrame
    df_sub = pd.DataFrame({"segment_id": segment_ids, "time_to_eruption": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save to CSV
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Print first few rows for verification
    print("Submission Head:")
    print(df_sub.head())
