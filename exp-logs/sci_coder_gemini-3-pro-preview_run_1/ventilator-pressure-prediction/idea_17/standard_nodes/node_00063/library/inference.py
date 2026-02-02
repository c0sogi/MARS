import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, get_device, Logger
from library.dataset import prepare_data
from library.model import WideStateNet


def predict(load_cached_data: bool = True):
    """
    Loads the trained model and generates predictions for the test set.
    Saves the submission file to Config.SUBMISSION_PATH.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
                                 Defaults to True.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    logger = Logger("inference_log.txt")

    logger.log("=== Starting Inference ===")

    # 2. Data Loading
    # We rely on the library function which handles caching and preprocessing.
    # prepare_data returns: train_loader, val_loader, test_loader, feature_names
    logger.log("Loading test data...")
    _, _, test_loader, feature_names = prepare_data(load_cached_data=load_cached_data)

    input_dim = len(feature_names)
    logger.log(f"Number of input features: {input_dim}")

    # 3. Model Initialization
    logger.log("Initializing model...")
    model = WideStateNet(input_dim=input_dim, feature_names=feature_names)
    model = model.to(device)

    # 4. Load Weights
    model_path = os.path.join(Config.WORKING_DIR, "model.pth")
    if not os.path.exists(model_path):
        error_msg = (
            f"Model checkpoint not found at {model_path}. Please train the model first."
        )
        logger.log(error_msg)
        raise FileNotFoundError(error_msg)

    logger.log(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    # 5. Inference Loop
    model.eval()
    predictions = []
    row_ids = []

    logger.log("Running prediction loop on test set...")

    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to device
            x = batch["x"].to(device)
            u_out = batch["u_out"].to(device)

            # IDs are needed for submission mapping (keep on CPU)
            ids = batch["ids"]

            # Forward pass
            # Model returns (final_pred, aux_pred) - we only need final_pred
            final_pred, _ = model(x, u_out)

            # Flatten the batch: (B, 80) -> (B*80,)
            # We must ensure we flatten consistently for both preds and ids
            preds_flat = final_pred.cpu().numpy().flatten()
            ids_flat = ids.numpy().flatten()

            predictions.append(preds_flat)
            row_ids.append(ids_flat)

    # 6. Post-processing
    logger.log("Aggregating predictions...")
    all_preds = np.concatenate(predictions)
    all_ids = np.concatenate(row_ids)

    # Create DataFrame
    submission_df = pd.DataFrame({"id": all_ids, "pressure": all_preds})

    # Ensure sorted by ID (standard submission format)
    submission_df = submission_df.sort_values("id")

    # 7. Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    save_path = Config.SUBMISSION_PATH

    logger.log(f"Saving submission to {save_path}...")
    submission_df.to_csv(save_path, index=False)

    logger.log("Inference complete.")
    logger.log(f"Generated predictions for {len(submission_df)} rows.")
    logger.close()
