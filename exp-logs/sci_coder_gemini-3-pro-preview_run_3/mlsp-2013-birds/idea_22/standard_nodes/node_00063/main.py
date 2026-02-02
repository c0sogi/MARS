import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.training import train_fold
from library.inference import predict_and_submit
from library.data import get_dataloaders
from library.models import get_model
from library.utils import seed_everything, calculate_roc_auc, get_device


def validate_ensemble(config):
    """
    Performs inference on the validation set using the trained ensemble
    and computes the final validation metric and failure analysis.
    """
    device = get_device()
    _, val_loader, _ = get_dataloaders(config, load_cached_data=True)

    # 1. Collect Ground Truth
    all_targets = []
    # We need to iterate the loader to get targets aligned with how we will predict
    for _, labels in val_loader:
        all_targets.append(labels.numpy())
    all_targets = np.concatenate(all_targets)

    # 2. Ensemble Inference
    num_samples = len(all_targets)
    num_classes = config.NUM_CLASSES
    ensemble_probs = np.zeros((num_samples, num_classes), dtype=np.float64)
    model_count = 0

    # Iterate over all Architectures, Folds, and Snapshot Ranks
    for arch in config.ARCHITECTURES:
        for fold in range(config.NUM_FOLDS):
            for rank in range(config.TOP_K_CHECKPOINTS):
                checkpoint_path = config.get_checkpoint_path(arch, fold, rank)

                if not os.path.exists(checkpoint_path):
                    continue

                # Load Model
                model = get_model(arch, config, pretrained=False)
                try:
                    state_dict = torch.load(checkpoint_path, map_location=device)
                    model.load_state_dict(state_dict)
                except Exception:
                    continue

                model.eval()

                # Inference Loop
                preds_list = []
                with torch.no_grad():
                    for images, _ in val_loader:
                        images = images.to(device)
                        outputs = model(images)
                        probs = torch.sigmoid(outputs)
                        preds_list.append(probs.cpu().numpy())

                # Accumulate
                ensemble_probs += np.concatenate(preds_list, axis=0)
                model_count += 1

    # Average Predictions
    if model_count > 0:
        final_probs = ensemble_probs / model_count
    else:
        final_probs = np.zeros((num_samples, num_classes), dtype=np.float64)

    # 3. Compute Metric
    val_auc = calculate_roc_auc(all_targets, final_probs)

    # 4. Failure Analysis
    # Calculate Mean Absolute Error per sample
    # Shape: (N_samples,)
    sample_errors = np.mean(np.abs(all_targets - final_probs), axis=1)

    # Extract Input Features from Dataset (stored in memory)
    # val_loader.dataset is a BirdDataset, which has .images attribute (numpy array)
    val_images = val_loader.dataset.images  # Shape: (N, H, W, 3)

    # Feature 1: Brightness (Mean pixel intensity)
    brightness = np.mean(val_images, axis=(1, 2, 3))

    # Feature 2: Contrast (Standard deviation of pixel intensity)
    contrast = np.std(val_images, axis=(1, 2, 3))

    # Compute Correlations
    # Handle edge cases where std is 0
    if np.std(sample_errors) > 0 and np.std(brightness) > 0:
        corr_brightness = np.corrcoef(sample_errors, brightness)[0, 1]
    else:
        corr_brightness = 0.0

    if np.std(sample_errors) > 0 and np.std(contrast) > 0:
        corr_contrast = np.corrcoef(sample_errors, contrast)[0, 1]
    else:
        corr_contrast = 0.0

    correlations = {"brightness": corr_brightness, "contrast": corr_contrast}

    return val_auc, correlations


def run():
    # 1. Configuration
    # Set epochs to 20 for a fast baseline that ensures convergence on this small dataset.
    config = Config(num_epochs=20, batch_size=32)
    seed_everything(config.SEED)

    # 2. Training Loop
    # Train heterogeneous ensemble: 3 Architectures x 5 Folds
    print("Starting Training Phase...")
    for arch in config.ARCHITECTURES:
        for fold in range(config.NUM_FOLDS):
            train_fold(config, fold, arch)

    # 3. Validation & Failure Analysis
    print("Starting Validation Phase...")
    val_auc, correlations = validate_ensemble(config)

    # REQUIRED OUTPUT: Final Validation Metric
    print(f"Final Validation Metric: {val_auc}")

    # REQUIRED OUTPUT: Failure Analysis
    print("Failure Analysis (Correlation of Error with Input Features):")
    for feature, corr in correlations.items():
        print(f"  Correlation with {feature}: {corr:.6f}")

    # 4. Submission Logic
    THRESHOLD = 0.9479806884980326

    if val_auc > THRESHOLD:
        print(
            f"Validation AUC ({val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(config)
    else:
        print(
            f"Validation AUC ({val_auc}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    run()
