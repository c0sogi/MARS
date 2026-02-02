import os
import torch
import numpy as np
import pandas as pd
from library.config import WORKING_DIR, SUBMISSION_DIR, SEED, BATCH_SIZE, NUM_WORKERS
from library.utils import seed_everything, get_device, compute_mcc
from library.dataset import get_dataloaders, get_test_loader
from library.model import SEARVN
from library.train import get_vocab_sizes


def run_inference(model, loader, device):
    """
    Runs inference on a dataloader and returns probabilities, targets, and IDs.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): The data loader.
        device (torch.device): The computation device.

    Returns:
        tuple: (probabilities, targets, contact_ids)
    """
    model.eval()
    all_probs = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            x_kin = batch["X_kin"].to(device)
            x_vis = batch["X_vis"].to(device)
            x_cat = batch["X_cat"].to(device)
            y = batch["y"].to(device)
            ids = batch["contact_id"]

            logits = model(x_kin, x_vis, x_cat)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate results
    all_probs = np.concatenate(all_probs).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    return all_probs, all_targets, all_ids


def optimize_threshold(model_path, debug=False, load_cached_data=True):
    """
    Finds the optimal decision threshold using the validation set.

    Args:
        model_path (str): Path to the saved model weights.
        debug (bool): Whether to run in debug mode (smaller dataset).
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (best_threshold, best_mcc)
    """
    seed_everything(SEED)
    device = get_device()

    print("Loading validation data for threshold optimization...")
    # We only need the validation loader
    _, val_loader = get_dataloaders(
        debug=debug,
        load_cached_data=load_cached_data,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # Initialize model
    vocab_sizes = get_vocab_sizes()
    model = SEARVN(vocab_sizes=vocab_sizes).to(device)

    # Load weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")

    # Run inference
    print("Running inference on validation set...")
    probs, targets, _ = run_inference(model, val_loader, device)

    # Grid search for best threshold
    print("Optimizing threshold...")
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_mcc = -1.0
    best_thresh = 0.5

    for thresh in thresholds:
        preds = (probs > thresh).astype(int)
        mcc = compute_mcc(targets, preds)

        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = thresh

    print(f"Optimization Complete. Best Threshold: {best_thresh}, Best MCC: {best_mcc}")
    return best_thresh, best_mcc


def generate_predictions(model_path, threshold, debug=False, load_cached_data=True):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model_path (str): Path to the saved model weights.
        threshold (float): The decision threshold to use.
        debug (bool): Whether to run in debug mode.
        load_cached_data (bool): Whether to use cached data.
    """
    seed_everything(SEED)
    device = get_device()

    print("Loading test data for inference...")
    test_loader = get_test_loader(
        debug=debug,
        load_cached_data=load_cached_data,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # Initialize model
    vocab_sizes = get_vocab_sizes()
    model = SEARVN(vocab_sizes=vocab_sizes).to(device)

    # Load weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")

    # Run inference
    print("Running inference on test set...")
    probs, _, ids = run_inference(model, test_loader, device)

    # Apply threshold
    preds = (probs > threshold).astype(int)

    # Create submission DataFrame
    submission_df = pd.DataFrame({"contact_id": ids, "contact": preds})

    # Ensure output directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"Positive predictions: {submission_df['contact'].sum()}")
