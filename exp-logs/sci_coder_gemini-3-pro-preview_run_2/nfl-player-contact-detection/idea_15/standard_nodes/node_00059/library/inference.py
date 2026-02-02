import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef
from library.config import Config
from library.model import CA_WRN
from library.data_processing import get_data_loaders, process_test_data


def optimize_threshold(model, val_loader, device):
    """
    Performs a grid search on the validation set to find the optimal decision threshold
    that maximizes the Matthews Correlation Coefficient (MCC).

    Args:
        model (nn.Module): The trained model.
        val_loader (DataLoader): DataLoader for the validation set.
        device (torch.device): Computation device.

    Returns:
        float: The optimal threshold.
        float: The best MCC score achieved.
    """
    model.eval()
    all_probs = []
    all_targets = []

    print("Generating validation predictions for threshold optimization...")
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)

            # Forward pass (returns logits)
            logits = model(features)
            probs = torch.sigmoid(logits).view(-1).cpu().numpy()
            targets = labels.view(-1).cpu().numpy()

            all_probs.append(probs)
            all_targets.append(targets)

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    # Grid search for best threshold
    thresholds = np.linspace(0, 1, Config.THRESHOLD_SEARCH_STEPS)
    best_mcc = -1.0
    best_threshold = 0.5

    print(f"Searching {len(thresholds)} thresholds...")

    for thresh in thresholds:
        preds = (all_probs >= thresh).astype(int)

        # Handle edge case where predictions are constant
        if len(np.unique(preds)) < 2:
            mcc = 0.0
        else:
            mcc = matthews_corrcoef(all_targets, preds)

        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = thresh

    print(f"Threshold Optimization Complete.")
    print(f"Best Threshold: {best_threshold}")
    print(f"Best Validation MCC: {best_mcc}")

    return best_threshold, best_mcc


def generate_submission(model, scaler, threshold, device):
    """
    Generates predictions for the test set using the optimized threshold and saves
    the submission file.

    Args:
        model (nn.Module): The trained model.
        scaler (StandardScaler): The fitted scaler from training.
        threshold (float): The optimized decision threshold.
        device (torch.device): Computation device.
    """
    print("Processing test data...")
    test_loader, contact_ids = process_test_data(scaler)

    model.eval()
    all_preds = []

    print("Running inference on test set...")
    with torch.no_grad():
        for features in test_loader:
            features = features.to(device)

            logits = model(features)
            probs = torch.sigmoid(logits).view(-1).cpu().numpy()

            # Apply threshold
            preds = (probs >= threshold).astype(int)
            all_preds.append(preds)

    all_preds = np.concatenate(all_preds)

    # Verify lengths match
    if len(all_preds) != len(contact_ids):
        raise ValueError(
            f"Length mismatch: {len(all_preds)} predictions vs {len(contact_ids)} IDs"
        )

    # Create submission DataFrame
    submission_df = pd.DataFrame({"contact_id": contact_ids, "contact": all_preds})

    output_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Submission shape: {submission_df.shape}")
    print(f"Predicted positive contacts: {submission_df['contact'].sum()}")


def run_inference():
    """
    Main entry point for the inference pipeline.
    Orchestrates data loading, model loading, threshold optimization, and submission generation.
    """
    # Set seeds
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference running on device: {device}")

    # 1. Load Validation Data & Metadata
    # We need the scaler and center_indices to reconstruct the model correctly
    print("Loading metadata and validation set...")
    _, val_loader, center_indices, scaler = get_data_loaders(load_cached_data=True)

    # Determine input dimension from a sample
    sample_features, _ = next(iter(val_loader))
    input_dim = sample_features.shape[1]
    print(f"Input Dimension: {input_dim}")
    print(f"Center Indices: {center_indices}")

    # 2. Initialize Model & Load Weights
    model = CA_WRN(
        input_dim=input_dim,
        center_indices=center_indices,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
    ).to(device)

    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run training first."
        )

    print(f"Loading model weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 3. Optimize Threshold
    best_threshold, best_mcc = optimize_threshold(model, val_loader, device)

    # 4. Generate Submission
    generate_submission(model, scaler, best_threshold, device)
