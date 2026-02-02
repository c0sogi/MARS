import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from library.config import Config
from library.feature_extractor import extract_features
from library.model import VolcanoMLP, set_seed


def generate_predictions(
    model=None,
    device=None,
    debug_size=None,
    load_cached_data=True,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
):
    """
    Generates predictions for the test set using the trained model and saved scaler.

    Args:
        model (nn.Module, optional): Pre-loaded model instance. If None, loads from disk.
        device (torch.device, optional): Device to perform inference on.
        debug_size (int, optional): Limit the number of test samples for debugging.
        load_cached_data (bool): Whether to attempt loading features from cache.
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of worker processes for data loading.

    Returns:
        pd.DataFrame: The submission DataFrame containing segment_id and predictions.
    """
    # 1. Configuration and Setup
    if device is None:
        device = torch.device(Config.DEVICE)

    set_seed(Config.SEED)

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print("Starting inference pipeline...")

    # 2. Load Test Features
    # extract_features handles the caching logic (load parquet if exists, else process)
    df_test = extract_features(
        Config.TEST_METADATA_PATH,
        Config.TEST_FEATURES_CACHE,
        load_cached_data=load_cached_data,
        debug_size=debug_size,
    )

    # 3. Prepare Feature Matrix
    # Identify feature columns (exclude metadata)
    exclude_cols = ["segment_id", "time_to_eruption"]
    feature_cols = [c for c in df_test.columns if c not in exclude_cols]

    X_test = df_test[feature_cols].values.astype(np.float32)
    segment_ids = df_test["segment_id"].values
    input_dim = len(feature_cols)

    # 4. Scale Features
    # Load scaler parameters (mean and scale) saved during the training phase
    if os.path.exists(Config.SCALER_MEAN_PATH) and os.path.exists(
        Config.SCALER_SCALE_PATH
    ):
        try:
            mean = np.load(Config.SCALER_MEAN_PATH)
            scale = np.load(Config.SCALER_SCALE_PATH)

            # Verify dimensions match to prevent broadcasting errors
            if len(mean) == input_dim:
                X_test = ((X_test - mean) / scale).astype(np.float32)
            else:
                print(
                    f"Warning: Scaler dimension ({len(mean)}) does not match input dimension ({input_dim}). "
                    "Skipping scaling. Check if feature extraction logic has changed."
                )
        except Exception as e:
            print(f"Error loading scaler: {e}. Proceeding with unscaled data.")
    else:
        print(
            f"Warning: Scaler files not found in {Config.WORKING_DIR}. "
            "Predictions will be generated on unscaled data."
        )

    # 5. Load Model
    if model is None:
        # Instantiate model structure
        model = VolcanoMLP(
            input_dim=input_dim,
            hidden_layers=Config.HIDDEN_LAYERS,
            dropout_rate=Config.DROPOUT_RATE,
        ).to(device)

        # Load weights
        if os.path.exists(Config.MODEL_SAVE_PATH):
            state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Loaded model weights from {Config.MODEL_SAVE_PATH}")
        else:
            print(
                f"Warning: Model weights not found at {Config.MODEL_SAVE_PATH}. "
                "Using randomly initialized weights (predictions will be random)."
            )
    else:
        model = model.to(device)

    # 6. Run Inference
    model.eval()
    predictions = []

    test_dataset = TensorDataset(torch.tensor(X_test))
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].to(device)
            outputs = model(inputs).squeeze()

            # Handle case where batch size is 1 or output is scalar
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)

            predictions.extend(outputs.cpu().numpy())

    # 7. Inverse Transform Predictions (Cite solution_lesson_node_00001)
    if os.path.exists(Config.TARGET_MEAN_PATH) and os.path.exists(
        Config.TARGET_STD_PATH
    ):
        t_mean = np.load(Config.TARGET_MEAN_PATH)
        t_std = np.load(Config.TARGET_STD_PATH)
        predictions = np.array(predictions) * t_std + t_mean
    else:
        print("Warning: Target scaler not found. Predictions are in scaled space.")

    # 8. Generate Submission File
    df_sub = pd.DataFrame({"segment_id": segment_ids, "time_to_eruption": predictions})

    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    return df_sub
