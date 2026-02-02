import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything
from library.data import get_test_dataloader
from library.model import SCVRNet


def run_inference(
    model_path="./working/best_model.pth", output_path=None, device_name=None
):
    """
    Runs inference on the test set using the SCVR-Net model and generates a submission file.

    Args:
        model_path (str): Path to the trained model weights.
        output_path (str): Path to save the submission CSV. If None, uses Config default.
        device_name (str): Device to run inference on ('cuda' or 'cpu'). If None, uses Config default.
    """
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)

    if device_name is None:
        device_name = Config.DEVICE
    device = torch.device(device_name)

    if output_path is None:
        output_path = Config.SUBMISSION_FILE

    print(f"Running inference using model: {model_path}")
    print(f"Device: {device}")

    # 2. Data
    # get_test_dataloader uses Config.TEST_CSV which is prepared by metadata script
    # and LungDataset which handles image loading and caching.
    test_loader = get_test_dataloader()
    print(f"Test batches: {len(test_loader)}")

    # 3. Model
    model = SCVRNet()

    # Load weights
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Model path {model_path} does not exist. Using random weights for debugging/demo."
        )

    model = model.to(device)
    model.eval()

    # 4. Inference Loop
    results = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            # Move inputs to device
            img_ax = data["img_ax"].to(device)
            img_cor = data["img_cor"].to(device)
            tabular = data["tabular"].to(device)

            # Metadata for reconstruction
            week_delta = data["week_delta"].to(device)
            baseline_fvc = data["baseline_fvc"].to(device)
            patient_weeks = data["patient_week"]  # List of strings

            # Forward pass
            # Output: (B, 3) -> [alpha, sigma_base, sigma_growth]
            params = model(img_ax, img_cor, tabular)

            alpha = params[:, 0]
            sigma_base = params[:, 1]
            sigma_growth = params[:, 2]

            # Parametric Prediction Logic
            # FVC = Baseline + alpha * delta_t
            fvc_pred = baseline_fvc + alpha * week_delta

            # Confidence = Sigma_base + Sigma_growth * |delta_t|
            sigma_pred = sigma_base + sigma_growth * torch.abs(week_delta)

            # Move to CPU for storage
            fvc_pred_np = fvc_pred.cpu().numpy()
            sigma_pred_np = sigma_pred.cpu().numpy()

            # Collect results
            for i in range(len(patient_weeks)):
                pw = patient_weeks[i]
                fvc = float(fvc_pred_np[i])
                sigma = float(sigma_pred_np[i])

                results.append({"Patient_Week": pw, "FVC": fvc, "Confidence": sigma})

    # 5. Post-processing and Saving
    df_sub = pd.DataFrame(results)

    # Apply Confidence Clipping (min 70) as per metric requirements
    # While the metric function handles this, submitting values < 70 is generally not useful.
    df_sub["Confidence"] = df_sub["Confidence"].clip(lower=70.0)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Total predictions: {len(df_sub)}")

    return df_sub
