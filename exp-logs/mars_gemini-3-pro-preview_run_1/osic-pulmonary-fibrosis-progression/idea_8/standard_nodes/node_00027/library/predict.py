import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.model import ResidualCrossAttentionNet
from library.data import get_dataloaders
from library.utils import seed_everything


def generate_predictions(weights_path=None, batch_size=None):
    """
    Generates predictions for the test set using the trained model.

    Args:
        weights_path (str, optional): Path to the model weights.
                                      Defaults to Config.WORKING_DIR/best_model.pth.
        batch_size (int, optional): Batch size for inference.
                                    Defaults to Config.BATCH_SIZE.

    Returns:
        pd.DataFrame: The submission dataframe containing Patient_Week, FVC, and Confidence.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    Config.setup_directories()

    if weights_path is None:
        weights_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if batch_size is not None:
        Config.BATCH_SIZE = batch_size

    print(f"Generating predictions using weights from: {weights_path}")

    # 2. Load Data
    # We only need the test loader
    _, _, test_loader = get_dataloaders(Config)

    # 3. Load Model
    model = ResidualCrossAttentionNet()

    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Weights file not found at {weights_path}. Using random initialization (debug mode)."
        )

    model.to(device)
    model.eval()

    # 4. Inference Loop
    all_fvc_pred = []
    all_sigma_pred = []

    print("Starting inference...")

    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to device
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)
            relative_week = batch["relative_week"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)

            # Forward pass
            # Model returns: fvc_pred, confidence_pred
            fvc_pred, sigma_pred = model(
                tabular, img_ax, img_cor, relative_week, baseline_fvc
            )

            # Move to CPU and collect
            all_fvc_pred.append(fvc_pred.cpu().numpy())
            all_sigma_pred.append(sigma_pred.cpu().numpy())

    # Concatenate results
    y_pred_fvc = np.concatenate(all_fvc_pred)
    y_pred_sigma = np.concatenate(all_sigma_pred)

    # 5. Format Submission
    # Load test metadata to get the correct Patient_Week identifiers
    # The test_loader iterates sequentially over Config.TEST_CSV
    test_df = pd.read_csv(Config.TEST_CSV)

    # Safety check
    if len(test_df) != len(y_pred_fvc):
        print(
            f"Critical Warning: Number of predictions ({len(y_pred_fvc)}) does not match metadata length ({len(test_df)})."
        )

    submission = pd.DataFrame(
        {
            "Patient_Week": test_df["Patient_Week"],
            "FVC": y_pred_fvc,
            "Confidence": y_pred_sigma,
        }
    )

    # Save submission
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    return submission
