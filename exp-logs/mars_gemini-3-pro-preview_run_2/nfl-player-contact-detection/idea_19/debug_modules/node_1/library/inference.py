import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, optimize_mcc_threshold
from library.data_processing import process_data
from library.dataset import get_dataloader
from library.models import SRVNet


def run_inference(model, dataloader, device):
    """
    Runs inference on a dataloader and returns probabilities and contact IDs.
    """
    model.eval()
    all_logits = []

    # We don't strictly need targets for test inference, but the dataloader yields them
    # IDs are needed to map predictions back to contact_ids

    with torch.no_grad():
        for x_kin, x_vis, _ in dataloader:
            x_kin = x_kin.to(device)
            x_vis = x_vis.to(device)

            logits = model(x_kin, x_vis)
            all_logits.append(logits.cpu())

    # Concatenate logits
    all_logits = torch.cat(all_logits).numpy().flatten()

    # Convert logits to probabilities: sigmoid(x) = 1 / (1 + exp(-x))
    probs = 1.0 / (1.0 + np.exp(-all_logits))

    return probs


def generate_predictions(load_cached_data: bool = True):
    """
    Main inference pipeline:
    1. Calibrate threshold using Validation Data.
    2. Generate predictions on Test Data.
    3. Save submission file.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference using device: {device}")

    # =========================================================================
    # 1. Model Loading
    # =========================================================================
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Please train first."
        )

    print(f"Loading model from {model_path}...")
    model = SRVNet(
        input_dim_kin=Config.INPUT_DIM_KINEMATIC,
        input_dim_vis=Config.INPUT_DIM_VISUAL,
        kinematic_hidden_dims=Config.KINEMATIC_HIDDEN_DIMS,
        visual_hidden_dims=Config.VISUAL_HIDDEN_DIMS,
        dropout_rate=Config.DROPOUT_RATE,
        lambda_visual=Config.LAMBDA_VISUAL,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # =========================================================================
    # 2. Calibration (Find Optimal Threshold on Validation Set)
    # =========================================================================
    print("Loading Validation Data for calibration...")
    # We need y_val to calculate MCC
    X_kin_val, X_vis_val, y_val, _ = process_data(
        "validation", load_cached_data=load_cached_data
    )

    val_loader = get_dataloader(
        X_kin_val,
        X_vis_val,
        y_val,
        batch_size=Config.BATCH_SIZE * 2,  # Larger batch size for inference
        shuffle=False,
    )

    print("Running inference on Validation set...")
    val_probs = run_inference(model, val_loader, device)

    print("Optimizing decision threshold...")
    best_threshold, best_mcc = optimize_mcc_threshold(y_val, val_probs)
    print(
        f"Calibration Complete. Best Threshold: {best_threshold}, Validation MCC: {best_mcc}"
    )

    # =========================================================================
    # 3. Test Inference
    # =========================================================================
    print("Loading Test Data...")
    X_kin_test, X_vis_test, y_test, ids_test = process_data(
        "test", load_cached_data=load_cached_data
    )

    # Note: y_test contains placeholders (zeros)
    test_loader = get_dataloader(
        X_kin_test,
        X_vis_test,
        y_test,
        ids=ids_test,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
    )

    print("Running inference on Test set...")
    test_probs = run_inference(model, test_loader, device)

    # =========================================================================
    # 4. Submission Generation
    # =========================================================================
    print("Generating submission file...")

    # Apply threshold
    test_preds = (test_probs >= best_threshold).astype(int)

    submission_df = pd.DataFrame({"contact_id": ids_test, "contact": test_preds})

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(f"Total predictions: {len(submission_df)}")
    print(
        f"Positive predictions: {submission_df['contact'].sum()} ({submission_df['contact'].mean():.4%})"
    )
