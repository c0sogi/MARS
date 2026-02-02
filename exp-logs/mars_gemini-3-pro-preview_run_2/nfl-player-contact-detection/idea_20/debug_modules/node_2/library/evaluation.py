import torch
import numpy as np
from sklearn.metrics import matthews_corrcoef
from library import config, dataset, models, data_processing
import os


def run_inference(model, loader, device):
    """
    Runs inference on a given loader and returns probabilities.

    Args:
        model: The PyTorch model.
        loader: DataLoader.
        device: Torch device.

    Returns:
        np.ndarray: Flattened array of probabilities.
        np.ndarray: Flattened array of targets (if available, else None).
    """
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for x_kin, x_vis, y in loader:
            x_kin = x_kin.to(device)
            x_vis = x_vis.to(device)

            logits = model(x_kin, x_vis)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(y.numpy())

    flat_probs = np.vstack(all_probs).flatten()
    flat_targets = np.concatenate(all_targets)

    return flat_probs, flat_targets


def optimize_threshold(model, val_loader, device):
    """
    Finds the threshold that maximizes MCC on the validation set.

    Args:
        model: The trained PyTorch model.
        val_loader: Validation DataLoader.
        device: Torch device.

    Returns:
        float: Best threshold.
        float: Best MCC score.
    """
    print("Running inference on validation set for threshold optimization...")
    probs, targets = run_inference(model, val_loader, device)

    best_mcc = -1.0
    best_threshold = 0.5

    # Grid search for threshold
    thresholds = np.arange(0.01, 1.00, 0.01)

    for t in thresholds:
        preds = (probs >= t).astype(int)
        mcc = matthews_corrcoef(targets, preds)

        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = t

    print(f"Threshold Optimization Results:")
    print(f"Best Threshold: {best_threshold}")
    print(f"Best Validation MCC: {best_mcc}")

    return best_threshold, best_mcc


def generate_predictions(model_path=None):
    """
    Main evaluation pipeline:
    1. Loads the best model.
    2. Optimizes threshold on validation set.
    3. Generates predictions on test set.
    4. Saves submission file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation running on device: {device}")

    if model_path is None:
        model_path = config.MODEL_SAVE_PATH

    # 1. Load Model
    print(f"Loading model from {model_path}...")
    model = models.GRVNet()

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Train the model first."
        )

    model.to(device)

    # 2. Optimize Threshold
    # We need the validation loader
    _, val_loader = dataset.get_train_val_loaders()
    best_threshold, best_val_mcc = optimize_threshold(model, val_loader, device)

    # 3. Test Inference
    print("Running inference on test set...")
    test_loader, test_ids = dataset.get_test_loader()
    test_probs, _ = run_inference(model, test_loader, device)

    # 4. Save Submission
    print(f"Generating submission with optimized threshold: {best_threshold}")
    data_processing.save_submission(test_ids, test_probs, threshold=best_threshold)

    return best_val_mcc
