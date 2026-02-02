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


from library.data import get_data


def validate_ensemble(config):
    """
    Performs Out-Of-Fold (OOF) inference using the trained ensemble
    and computes the final validation metric and failure analysis.
    """
    device = get_device()

    # Load full dev set to initialize arrays
    (dev_imgs, dev_lbls), _ = get_data(config, load_cached_data=True)
    num_samples = len(dev_imgs)
    num_classes = config.NUM_CLASSES

    oof_probs = np.zeros((num_samples, num_classes), dtype=np.float64)
    oof_counts = np.zeros((num_samples, num_classes), dtype=np.float64)

    # Iterate over folds to generate predictions for the validation part of that fold
    for fold in range(config.NUM_FOLDS):
        print(f"Validating Fold {fold}...")
        # Get validation loader for this fold
        _, val_loader, _, val_idx = get_dataloaders(
            config, fold=fold, load_cached_data=True
        )

        # Identify models trained on this fold (Arch x Rank)
        fold_probs = np.zeros((len(val_idx), num_classes), dtype=np.float64)
        model_count = 0

        for arch in config.ARCHITECTURES:
            for rank in range(config.TOP_K_CHECKPOINTS):
                checkpoint_path = config.get_checkpoint_path(arch, fold, rank)

                if not os.path.exists(checkpoint_path):
                    continue

                model = get_model(arch, config, pretrained=False)
                try:
                    state_dict = torch.load(checkpoint_path, map_location=device)
                    model.load_state_dict(state_dict)
                except Exception:
                    continue

                model.eval()

                preds_list = []
                with torch.no_grad():
                    for images, _ in val_loader:
                        images = images.to(device)
                        outputs = model(images)
                        probs = torch.sigmoid(outputs)
                        preds_list.append(probs.cpu().numpy())

                if preds_list:
                    fold_probs += np.concatenate(preds_list, axis=0)
                    model_count += 1

        if model_count > 0:
            fold_probs /= model_count
            oof_probs[val_idx] += fold_probs
            oof_counts[val_idx] += 1

    valid_mask = oof_counts > 0
    final_probs = np.zeros_like(oof_probs)
    final_probs[valid_mask] = oof_probs[valid_mask] / oof_counts[valid_mask]

    # 3. Compute Metric
    val_auc = calculate_roc_auc(dev_lbls, final_probs)

    # 4. Failure Analysis
    sample_errors = np.mean(np.abs(dev_lbls - final_probs), axis=1)

    # Features
    brightness = np.mean(dev_imgs, axis=(1, 2, 3))
    contrast = np.std(dev_imgs, axis=(1, 2, 3))

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
    # Increase epochs to 40 for better convergence with K-Fold
    config = Config(num_epochs=40, batch_size=32)
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
