import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import NSLHN


def inference(model_path=None, batch_size=None, num_workers=None, debug=False):
    """
    Performs inference on the test set using the NSL-HN model and generates the submission file.

    Args:
        model_path (str, optional): Path to the trained model weights. Defaults to Config.MODEL_SAVE_PATH.
        batch_size (int, optional): Batch size for inference. Defaults to Config.BATCH_SIZE.
        num_workers (int, optional): Number of worker threads for data loading. Defaults to Config.NUM_WORKERS.
        debug (bool, optional): If True, runs inference on a small subset of the test data.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    if model_path is None:
        model_path = Config.MODEL_SAVE_PATH

    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    if num_workers is None:
        num_workers = Config.NUM_WORKERS

    print(f"Starting inference with model: {model_path}")
    print(f"Device: {device}")

    # 2. Data Loading
    # We only need the test loader. get_dataloaders returns (train, val, test)
    _, _, test_loader = get_dataloaders(batch_size=batch_size, num_workers=num_workers)

    # Load test metadata to get the Patient_Week identifiers
    # The test_loader iterates sequentially over Config.TEST_CSV, so orders align.
    try:
        test_df = pd.read_csv(Config.TEST_CSV)
        patient_weeks = test_df["Patient_Week"].values
    except FileNotFoundError:
        print(f"Error: Test metadata file not found at {Config.TEST_CSV}")
        return

    # 3. Model Initialization
    model = NSLHN()
    model = model.to(device)

    # 4. Load Weights
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    else:
        print(
            f"WARNING: Model path {model_path} does not exist. Using random weights (for testing only)."
        )

    model.eval()

    # 5. Prediction Loop
    pred_fvcs = []
    pred_sigmas = []

    print(f"Processing {len(test_loader)} batches...")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if debug and i >= 5:
                print("Debug mode: stopping after 5 batches.")
                break

            # Move inputs to device
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            rel_week = batch["relative_week"].to(device)
            base_fvc = batch["baseline_fvc"].to(device)

            # Forward pass
            # The model forward method handles the parametric logic:
            # FVC = Baseline + alpha * week
            # Sigma = Base + Growth * |week|
            fvc, sigma = model(img_ax, img_cor, tabular, rel_week, base_fvc)

            pred_fvcs.extend(fvc.cpu().numpy())
            pred_sigmas.extend(sigma.cpu().numpy())

    # 6. Post-processing
    pred_fvcs = np.array(pred_fvcs)
    pred_sigmas = np.array(pred_sigmas)

    # Clip confidence values at 70ml as per metric requirements
    # "confidence values are clipped at 70 ml to reflect the approximate measurement uncertainty"
    pred_sigmas = np.maximum(pred_sigmas, 70.0)

    # Handle debug case truncation for IDs
    if len(pred_fvcs) != len(patient_weeks):
        if debug:
            patient_weeks = patient_weeks[: len(pred_fvcs)]
        else:
            print(
                f"Error: Mismatch between predictions ({len(pred_fvcs)}) and test set size ({len(patient_weeks)})."
            )
            # In a real scenario, we might raise an error, but here we truncate to match to allow saving
            min_len = min(len(pred_fvcs), len(patient_weeks))
            pred_fvcs = pred_fvcs[:min_len]
            pred_sigmas = pred_sigmas[:min_len]
            patient_weeks = patient_weeks[:min_len]

    # 7. Generate Submission
    submission = pd.DataFrame(
        {"Patient_Week": patient_weeks, "FVC": pred_fvcs, "Confidence": pred_sigmas}
    )

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save to CSV
    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")

    # Print sample
    print("\nSample predictions:")
    print(submission.head())
