import torch
import numpy as np
from library.config import Config
from library.model import WideSERepNeXt
from library.utils import load_checkpoint, save_submission
from library.dataset import get_dataloaders


def load_and_reparameterize_model(seed, device):
    """
    Loads a model checkpoint and converts it to inference mode by fusing branches.

    Args:
        seed (int): The random seed associated with the model checkpoint.
        device (str): The device to load the model onto.

    Returns:
        model (nn.Module): The re-parameterized model in eval mode, or None if checkpoint missing.
    """
    # Initialize model in training mode structure (multi-branch) to load weights correctly
    model = WideSERepNeXt(deploy=False)
    model = model.to(device)

    # Load weights
    try:
        model = load_checkpoint(model, seed, device)
    except FileNotFoundError:
        print(f"Warning: Checkpoint for seed {seed} not found.")
        return None

    # Switch to deploy mode (structural re-parameterization)
    # This fuses the parallel branches (3x3, 1x1, Identity) into single 3x3 convolutions
    model.reparameterize()
    model.eval()

    return model


def predict_with_tta(model, loader, device):
    """
    Performs inference with Test Time Augmentation (Original, H-Flip, V-Flip).

    Args:
        model (nn.Module): The trained and re-parameterized model.
        loader (DataLoader): DataLoader for the test set.
        device (str): Computation device.

    Returns:
        np.array: 1D array of predicted probabilities.
    """
    preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # 1. Original
            out_orig = model(images)
            prob_orig = torch.sigmoid(out_orig)

            # 2. Horizontal Flip (dim 3 is width)
            images_h = torch.flip(images, [3])
            out_h = model(images_h)
            prob_h = torch.sigmoid(out_h)

            # 3. Vertical Flip (dim 2 is height)
            images_v = torch.flip(images, [2])
            out_v = model(images_v)
            prob_v = torch.sigmoid(out_v)

            # Average TTA probabilities
            avg_prob = (prob_orig + prob_h + prob_v) / 3.0
            preds.append(avg_prob.cpu().numpy())

    return np.concatenate(preds).flatten()


def run_inference(seeds=Config.SEEDS, load_cached_data=True):
    """
    Main driver for inference. Loads data, runs ensemble inference with TTA,
    averages predictions, and saves the submission.

    Args:
        seeds (list): List of seeds to use for the ensemble.
        load_cached_data (bool): Whether to use cached dataset files.
    """
    device = Config.DEVICE

    # Load data
    # We only need test_loader and test_ids
    _, _, test_loader, test_ids = get_dataloaders(load_cached_data=load_cached_data)

    ensemble_preds = []

    print("Starting Inference with Ensemble and TTA...")

    for seed in seeds:
        print(f"Processing Seed {seed}...")

        # Load and optimize model structure
        model = load_and_reparameterize_model(seed, device)
        if model is None:
            continue

        # Generate predictions with TTA
        seed_preds = predict_with_tta(model, test_loader, device)
        ensemble_preds.append(seed_preds)

    if not ensemble_preds:
        print("Error: No predictions generated.")
        return

    # Homogeneous Seed Averaging
    # Average predictions across all valid seeds
    ensemble_preds = np.array(ensemble_preds)
    final_preds = np.mean(ensemble_preds, axis=0)

    # Save Submission
    save_submission(test_ids, final_preds)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
