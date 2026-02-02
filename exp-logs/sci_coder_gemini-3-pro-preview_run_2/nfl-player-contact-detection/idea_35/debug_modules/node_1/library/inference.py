import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import compute_mcc
from library.model import APIRVNet
from library.dataset import get_train_val_datasets, get_test_dataset


def optimize_threshold(model, val_loader, device):
    """
    Performs a grid search on the validation set to find the decision threshold
    that maximizes the Matthews Correlation Coefficient (MCC).

    Args:
        model (nn.Module): The trained APIRVNet model.
        val_loader (DataLoader): DataLoader for the validation set.
        device (torch.device): Device to run inference on.

    Returns:
        float: The optimized threshold.
        float: The best MCC score achieved.
    """
    model.eval()
    all_probs = []
    all_targets = []

    # 1. Collect Predictions
    with torch.no_grad():
        for inputs, targets in val_loader:
            x_kin, x_vis = inputs
            x_kin = x_kin.to(device, non_blocking=True)
            x_vis = x_vis.to(device, non_blocking=True)

            logits = model(x_kin, x_vis)
            probs = torch.sigmoid(logits).view(-1)

            all_probs.append(probs.cpu())
            all_targets.append(targets.view(-1).cpu())

    # Concatenate
    y_prob = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_targets).numpy()

    # 2. Grid Search
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_mcc = -1.0

    # Vectorized search is possible but loop is memory safe and fast enough for 100 steps
    for thresh in thresholds:
        y_pred = (y_prob > thresh).astype(np.float32)
        score = compute_mcc(y_true, y_pred)

        if score > best_mcc:
            best_mcc = score
            best_threshold = thresh

    print(
        f"Threshold Optimization Complete. Best Threshold: {best_threshold}, Best MCC: {best_mcc}"
    )

    return best_threshold, best_mcc


def generate_predictions(model, test_loader, ids, threshold, device):
    """
    Generates predictions for the test set using the optimized threshold
    and saves the submission file.

    Args:
        model (nn.Module): The trained APIRVNet model.
        test_loader (DataLoader): DataLoader for the test set.
        ids (np.ndarray): Array of contact_ids corresponding to the test set.
        threshold (float): Optimized decision threshold.
        device (torch.device): Device to run inference on.
    """
    model.eval()
    all_probs = []

    print("Generating predictions for test set...")

    with torch.no_grad():
        for inputs in test_loader:
            # Test loader might return (inputs) or (inputs, targets) depending on implementation
            # dataset.py __getitem__ returns inputs tuple if y is None
            if isinstance(inputs, list) or isinstance(inputs, tuple):
                # Check if it's (x_kin, x_vis) or ((x_kin, x_vis), target)
                # Based on dataset.py: returns inputs tuple (x_kin, x_vis) if y is None
                x_kin, x_vis = inputs
            else:
                raise ValueError("Unexpected input format from test loader")

            x_kin = x_kin.to(device, non_blocking=True)
            x_vis = x_vis.to(device, non_blocking=True)

            logits = model(x_kin, x_vis)
            probs = torch.sigmoid(logits).view(-1)
            all_probs.append(probs.cpu())

    # Concatenate probabilities
    y_probs = torch.cat(all_probs).numpy()

    # Apply Threshold
    y_preds = (y_probs > threshold).astype(int)

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"contact_id": ids, "contact": y_preds})

    # Save
    save_path = Config.SUBMISSION_PATH
    df_sub.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path} with {len(df_sub)} rows.")


def run_inference(debug=False):
    """
    Orchestrates the full inference pipeline:
    1. Loads Validation Data to optimize threshold.
    2. Loads Test Data.
    3. Loads Model.
    4. Optimizes Threshold.
    5. Generates Submission.

    Args:
        debug (bool): If True, uses a subset of data.
    """
    device = torch.device(Config.DEVICE)
    print(f"Running inference on device: {device}")

    # 1. Load Validation Data (for thresholding)
    print("Loading validation data for threshold optimization...")
    # We only need val dataset here, but get_train_val_datasets returns both
    _, val_dataset = get_train_val_datasets(debug=debug)

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Initialize Model
    # We need to know input dims from the dataset
    kin_dim = len(val_dataset.kin_indices)
    vis_dim = len(val_dataset.vis_indices)

    print(f"Initializing model with Kinematic Dim: {kin_dim}, Visual Dim: {vis_dim}")
    model = APIRVNet(kin_input_dim=kin_dim, vis_input_dim=vis_dim)
    model.to(device)

    # 3. Load Weights
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading weights from {Config.MODEL_SAVE_PATH}")
        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Model weights not found at {Config.MODEL_SAVE_PATH}. Using random initialization (expect poor results)."
        )

    # 4. Optimize Threshold
    best_threshold, best_mcc = optimize_threshold(model, val_loader, device)

    # 5. Load Test Data
    print("Loading test data...")
    test_dataset, test_ids = get_test_dataset()

    # Verify dimensions match
    test_kin_dim = len(test_dataset.kin_indices)
    test_vis_dim = len(test_dataset.vis_indices)

    if test_kin_dim != kin_dim or test_vis_dim != vis_dim:
        print(
            f"Warning: Test dimensions ({test_kin_dim}, {test_vis_dim}) do not match Train/Val dimensions ({kin_dim}, {vis_dim})."
        )
        # In a real scenario, this would likely crash the model forward pass or require handling.
        # Assuming prepare_features guarantees consistent columns via cache or logic.

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 6. Generate Predictions
    generate_predictions(model, test_loader, test_ids, best_threshold, device)
