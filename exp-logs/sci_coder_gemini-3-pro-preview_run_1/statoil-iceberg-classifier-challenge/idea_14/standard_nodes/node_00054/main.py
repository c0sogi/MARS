import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

from library.config import Config
from library.utils import seed_everything, load_model, log_message
from library.dataset import get_train_val_loaders
from library.model import IcebergResNet18
from library.engine import run_swa_training
from library.inference import predict_ensemble, create_submission


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    log_message("Initializing ResNet-18 SWA Ensemble Pipeline (Supervised)...")

    # Ensure directories exist
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Load Data
    # Using the fixed validation split defined in metadata
    train_loader, val_loader = get_train_val_loaders(load_cached_data=True)

    # 3. Ensemble Training
    # We train independent models on the labeled training set
    # Diversity comes from random initialization and data augmentation
    models = []
    n_models = Config.N_FOLDS  # Using N_FOLDS as the number of ensemble members

    log_message(f"\n=== Training {n_models} SWA Ensemble Models ===")

    for i in range(n_models):
        log_message(f"\nTraining Model {i+1}/{n_models}")

        # Initialize fresh model
        model = IcebergResNet18()

        # Define checkpoint paths
        save_path_best = os.path.join(Config.CHECKPOINT_DIR, f"model_{i}_best.pth")
        save_path_swa = os.path.join(Config.CHECKPOINT_DIR, f"model_{i}_swa.pth")

        # Run SWA Training
        run_swa_training(
            model,
            train_loader,
            val_loader,
            device=Config.DEVICE,
            save_path_best=save_path_best,
            save_path_swa=save_path_swa,
        )

        # Load the best available weights (SWA if available, else Best Standard)
        if os.path.exists(save_path_swa):
            model = load_model(model, save_path_swa, device=Config.DEVICE)
            log_message(f"Loaded SWA weights for Model {i+1}")
        else:
            model = load_model(model, save_path_best, device=Config.DEVICE)
            log_message(f"Loaded Best Standard weights for Model {i+1}")

        models.append(model)

    # 4. Final Validation & Metrics
    log_message("\n=== Final Validation ===")

    # Aggregate predictions on validation set
    # We need to manually iterate to get labels and features for failure analysis
    all_preds = []
    all_labels = []
    all_angles = []
    all_img_means = []

    # Set models to eval
    for m in models:
        m.eval()
        m.to(Config.DEVICE)

    with torch.no_grad():
        for images, angles, labels in val_loader:
            images = images.to(Config.DEVICE)
            angles_gpu = angles.to(Config.DEVICE)

            # Ensemble Prediction (with TTA inside predict_ensemble logic manually applied here)
            # Since predict_ensemble expects a test loader with IDs, we implement a simple ensemble loop here
            batch_preds = []

            # TTA Views
            images_h = torch.flip(images, dims=[3])
            images_v = torch.flip(images, dims=[2])

            for model in models:
                l_orig = model(images, angles_gpu)
                l_h = model(images_h, angles_gpu)
                l_v = model(images_v, angles_gpu)

                p_avg = (
                    torch.sigmoid(l_orig) + torch.sigmoid(l_h) + torch.sigmoid(l_v)
                ) / 3.0
                batch_preds.append(p_avg)

            # Average across models
            batch_preds = torch.stack(batch_preds).mean(dim=0)

            all_preds.extend(batch_preds.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())
            all_angles.extend(angles.numpy().flatten())

            # Calculate image mean for failure analysis (using channel 0 - HH)
            # images is (B, 3, 224, 224), channel 0 is HH
            img_means = images[:, 0, :, :].mean(dim=(1, 2)).cpu().numpy()
            all_img_means.extend(img_means)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate Log Loss
    # Clip predictions to avoid log(0) - sklearn does this but good to be explicit
    all_preds = np.clip(all_preds, 1e-15, 1 - 1e-15)
    final_metric = log_loss(all_labels, all_preds)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    log_message("\n=== Failure Analysis ===")
    errors = np.abs(all_labels - all_preds)

    # Correlation with Incidence Angle
    corr_angle, _ = pearsonr(errors, all_angles)
    print(f"Correlation (Error vs Inc Angle): {corr_angle:.4f}")

    # Correlation with Image Mean (Signal Strength)
    corr_signal, _ = pearsonr(errors, all_img_means)
    print(f"Correlation (Error vs Signal Strength): {corr_signal:.4f}")

    # 8. Submission
    threshold = 0.16918645240183008
    if final_metric < threshold:
        log_message(
            f"\nMetric ({final_metric}) is better than threshold ({threshold}). Generating submission..."
        )

        # Generate predictions on test set using the Ensemble
        # We pass the list of models to predict_ensemble which handles TTA
        from library.dataset import get_test_loader

        test_loader = get_test_loader(load_cached_data=True)

        test_predictions = predict_ensemble(models, test_loader, device=Config.DEVICE)

        create_submission(test_predictions)
    else:
        log_message(
            f"\nMetric ({final_metric}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
