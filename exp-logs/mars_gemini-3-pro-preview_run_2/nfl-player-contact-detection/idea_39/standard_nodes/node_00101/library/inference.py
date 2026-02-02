import os
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import compute_mcc
from library.model import predict


def optimize_threshold(y_true, y_probs, step=0.01):
    """
    Performs a grid search to find the optimal decision threshold that maximizes MCC.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth binary labels.
        y_probs (np.ndarray or torch.Tensor): Predicted probabilities.
        step (float): Step size for grid search.

    Returns:
        float: The optimal threshold.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_probs, torch.Tensor):
        y_probs = y_probs.detach().cpu().numpy()

    y_true = y_true.flatten()
    y_probs = y_probs.flatten()

    # Define search range
    thresholds = np.arange(0.05, 0.96, step)

    best_mcc = -1.0
    best_thresh = 0.5

    for t in thresholds:
        preds = (y_probs > t).astype(int)
        mcc = compute_mcc(y_true, preds)

        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = t

    print(f"Validation Best MCC: {best_mcc}")
    return best_thresh


def predict_and_submit(
    model, test_loader, device, threshold, save_path=None, metadata_df=None
):
    """
    Generates predictions for the test set using the optimized threshold and saves the results.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set.
        device (str): Device to run inference on.
        threshold (float): The decision threshold to apply.
        save_path (str, optional): Path to save the submission CSV. Defaults to Config.SUBMISSION_PATH.
        metadata_df (pd.DataFrame, optional): Metadata DataFrame containing contact_ids.
                                              If None, loads from Config.METADATA_DIR/test.csv.
    """
    if save_path is None:
        save_path = Config.SUBMISSION_PATH

    # 1. Generate Probabilities
    # Uses the predict function from library.model which returns a flattened numpy array
    probs = predict(model, test_loader, device)

    # 2. Apply Threshold
    preds = (probs > threshold).astype(int)

    # 3. Load Metadata
    if metadata_df is None:
        meta_path = os.path.join(Config.METADATA_DIR, "test.csv")
        if os.path.exists(meta_path):
            metadata_df = pd.read_csv(meta_path)
        else:
            raise FileNotFoundError(f"Test metadata not found at {meta_path}")

    # 4. Align Lengths
    # Handle cases where the test loader might be a subset (e.g., during debugging)
    # or metadata has more rows than predictions.
    n_preds = len(preds)
    n_meta = len(metadata_df)

    if n_preds != n_meta:
        min_len = min(n_preds, n_meta)
        preds = preds[:min_len]
        metadata_df = metadata_df.iloc[:min_len]

    # 5. Create Submission DataFrame
    submission = pd.DataFrame(
        {"contact_id": metadata_df["contact_id"], "contact": preds}
    )

    # 6. Save Submission
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    submission.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
