import os
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import load_checkpoint
from library.dataset import get_dataloaders
from library.model import MultiScaleNarrowResNet


def predict_with_tta(model, loader, device):
    """
    Generates predictions using Test Time Augmentation (TTA).
    TTA Strategy: Original + Horizontal Flip + Vertical Flip.

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): Test data loader.
        device (torch.device): Computation device.

    Returns:
        tuple: (predictions numpy array, ids list)
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)

            # 1. Original Prediction
            logits = model(images)
            prob_orig = torch.sigmoid(logits)

            # 2. Horizontal Flip Prediction
            images_h = torch.flip(images, dims=[3])
            logits_h = model(images_h)
            prob_h = torch.sigmoid(logits_h)

            # 3. Vertical Flip Prediction
            images_v = torch.flip(images, dims=[2])
            logits_v = model(images_v)
            prob_v = torch.sigmoid(logits_v)

            # Average TTA predictions
            avg_prob = (prob_orig + prob_h + prob_v) / 3.0

            all_preds.append(avg_prob.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate all batches
    # Shape will be (N, 1) or (N,) depending on squeeze
    predictions = np.concatenate(all_preds, axis=0)

    # Flatten to (N,)
    if predictions.ndim > 1:
        predictions = predictions.flatten()

    return predictions, all_ids


def generate_submission():
    """
    Loads the ensemble of models, performs inference with TTA,
    aggregates predictions, and saves the submission file.
    """
    print("Initializing Submission Generation...")

    # 1. Setup Device
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data
    # We only need the test loader
    _, _, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Ensemble Inference
    ensemble_preds = []
    final_ids = None

    for seed in Config.SEEDS:
        print(f"Processing model for seed {seed}...")

        # Initialize model
        model = MultiScaleNarrowResNet().to(device)

        # Load weights
        filename = f"model_seed_{seed}.pth"
        try:
            load_checkpoint(model, filename, device=device)
        except FileNotFoundError:
            print(f"--> Checkpoint {filename} not found. Skipping this seed.")
            continue

        # Predict with TTA
        preds, ids = predict_with_tta(model, test_loader, device)

        ensemble_preds.append(preds)

        # Capture IDs from the first successful run (order is deterministic)
        if final_ids is None:
            final_ids = ids

    if not ensemble_preds:
        raise RuntimeError(
            "No models were successfully loaded. Cannot generate submission."
        )

    # 4. Aggregate Predictions (Mean of Ensemble)
    # Stack to (Num_Seeds, N) then mean over axis 0
    ensemble_preds = np.vstack(ensemble_preds)
    final_predictions = np.mean(ensemble_preds, axis=0)

    print(f"Aggregated predictions from {len(ensemble_preds)} models.")

    # 5. Create Submission DataFrame
    df = pd.DataFrame({"id": final_ids, "has_cactus": final_predictions})

    # 6. Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved successfully to: {Config.SUBMISSION_PATH}")
    print(f"First 5 rows:\n{df.head()}")
