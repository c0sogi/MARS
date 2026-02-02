import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef

from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.data_processing import get_dataloaders
from library.models import SPIRVNet, FocalLoss
from library.train import validate, find_optimal_threshold


def optimize_threshold(model, val_loader, device):
    """
    Performs a grid search on validation probabilities to find the decision
    threshold that maximizes the Matthews Correlation Coefficient (MCC).

    Args:
        model: The trained PyTorch model.
        val_loader: DataLoader for the validation set.
        device: Computation device.

    Returns:
        float: The optimized threshold value.
        float: The best MCC score associated with that threshold.
    """
    # We need a criterion to use the existing validate function,
    # though we strictly only care about probs and targets here.
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)

    # Get probabilities and targets from the validation set
    # validate returns: avg_loss, all_probs, all_targets
    _, val_probs, val_targets = validate(model, val_loader, criterion, device)

    # Use the utility from library.train to find the best threshold
    best_thresh, best_mcc = find_optimal_threshold(val_targets, val_probs)

    return best_thresh, best_mcc


def predict_test_set(model, test_loader, device):
    """
    Generates probabilities for the test data using the trained model.

    Args:
        model: The trained PyTorch model.
        test_loader: DataLoader for the test set.
        device: Computation device.

    Returns:
        np.ndarray: Flattened array of predicted probabilities.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            # Test loader returns (kin, vis)
            kin, vis = batch
            kin = kin.to(device)
            vis = vis.to(device)

            logits = model(kin, vis)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    if len(all_probs) > 0:
        all_probs = np.concatenate(all_probs).flatten()
    else:
        all_probs = np.array([])

    return all_probs


def generate_submission(test_probs, threshold, output_path):
    """
    Applies the optimized threshold to test probabilities and formats
    the final CSV submission file.

    Args:
        test_probs (np.ndarray): Predicted probabilities for the test set.
        threshold (float): Decision threshold.
        output_path (str): Path to save the submission CSV.
    """
    # Load test metadata to ensure correct contact_id ordering
    df_test_meta = pd.read_csv(Config.METADATA_TEST)

    if len(df_test_meta) != len(test_probs):
        raise ValueError(
            f"Mismatch in prediction length: Metadata {len(df_test_meta)} vs Preds {len(test_probs)}"
        )

    # Apply threshold to generate binary predictions
    predictions = (test_probs >= threshold).astype(int)

    df_test_meta["contact"] = predictions

    # Select required columns
    submission_df = df_test_meta[["contact_id", "contact"]]

    # Save
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_inference(load_cached_data=True):
    """
    Orchestrates the inference pipeline:
    1. Loads data.
    2. Initializes model and loads weights.
    3. Optimizes threshold on validation set.
    4. Predicts on test set.
    5. Generates submission file.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Load Data
    # get_dataloaders returns train, val, test loaders
    _, val_loader, test_loader = get_dataloaders(load_cached_data=load_cached_data)

    # 2. Initialize Model
    # Determine input dimensions from a batch in the val loader
    dummy_kin, dummy_vis, _ = next(iter(val_loader))
    input_dim_kin = dummy_kin.shape[1]
    input_dim_vis = dummy_vis.shape[1]

    print(
        f"Initializing SPIRVNet with Kinematic Dim: {input_dim_kin}, Visual Dim: {input_dim_vis}"
    )
    model = SPIRVNet(input_dim_kin, input_dim_vis).to(device)

    # Load best model weights
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"
        )

    checkpoint = load_checkpoint(model, Config.MODEL_SAVE_PATH, device=Config.DEVICE)
    print(
        f"Loaded model from epoch {checkpoint['epoch']} with validation MCC {checkpoint['score']}"
    )

    # 3. Optimize Threshold
    print("Optimizing threshold on validation set...")
    best_thresh, best_mcc = optimize_threshold(model, val_loader, device)
    print(f"Validation Results - Best Threshold: {best_thresh}, Best MCC: {best_mcc}")

    # 4. Predict Test Set
    print("Generating predictions for test set...")
    test_probs = predict_test_set(model, test_loader, device)

    # 5. Generate Submission
    print(f"Generating submission with threshold {best_thresh}...")
    generate_submission(test_probs, best_thresh, Config.SUBMISSION_PATH)
