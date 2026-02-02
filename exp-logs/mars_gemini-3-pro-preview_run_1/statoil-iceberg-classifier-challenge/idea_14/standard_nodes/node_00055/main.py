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
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from library.dataset import (
    get_training_data_arrays,
    IcebergDataset,
    get_transforms,
    get_train_val_loaders,
)


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    log_message("Initializing ResNet-18 SWA Ensemble with Global Epoch Selection...")

    # Ensure directories exist
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Phase 1: Global Epoch Selection via 5-Fold CV
    # We use the training subset defined in metadata for CV
    log_message("\n=== Phase 1: Global Epoch Selection (5-Fold CV) ===")

    images_train, angles_train, labels_train = get_training_data_arrays(
        load_cached_data=True
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=Config.SEED)
    fold_histories = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(images_train)):
        log_message(f"\nRunning CV Fold {fold+1}/5")

        # Create DataLoaders for this fold
        ds_train = IcebergDataset(
            images_train[train_idx],
            angles_train[train_idx],
            labels_train[train_idx],
            transform=get_transforms("train"),
            mode="train",
        )
        ds_val = IcebergDataset(
            images_train[val_idx],
            angles_train[val_idx],
            labels_train[val_idx],
            transform=get_transforms("val"),
            mode="val",
        )

        dl_train = DataLoader(
            ds_train,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
        )
        dl_val = DataLoader(
            ds_val,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        model = IcebergResNet18()

        # Run training without early stopping to get full curve
        history = run_swa_training(
            model,
            dl_train,
            dl_val,
            device=Config.DEVICE,
            save_path_best=None,
            save_path_swa=None,
            early_stopping=False,
        )
        fold_histories.append(history["val_loss"])

    # Calculate optimal epoch
    # Average validation loss across folds for each epoch
    # Note: history lists are 0-indexed (epoch 1 is index 0)
    avg_val_losses = np.mean(fold_histories, axis=0)
    optimal_epoch_idx = np.argmin(avg_val_losses)
    optimal_epochs = optimal_epoch_idx + 1
    min_avg_loss = avg_val_losses[optimal_epoch_idx]

    log_message(f"\nGlobal Epoch Selection Results:")
    log_message(f"Optimal Epochs: {optimal_epochs}")
    log_message(f"Minimum Average CV Loss: {min_avg_loss:.4f}")

    # 3. Phase 2: Train Final Ensemble on Full Training Subset
    # We train 5 models on the full training data for the calibrated duration
    log_message(f"\n=== Phase 2: Training Final Ensemble ({optimal_epochs} epochs) ===")

    # Full training loader (no split)
    ds_full_train = IcebergDataset(
        images_train,
        angles_train,
        labels_train,
        transform=get_transforms("train"),
        mode="train",
    )
    dl_full_train = DataLoader(
        ds_full_train,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )

    # We still need a validation loader for run_swa_training, but we won't use it for stopping.
    # We can use the hold-out validation set just for logging.
    _, dl_holdout_val = get_train_val_loaders(load_cached_data=True)

    ensemble_models = []
    n_models = 5

    # Adjust SWA schedule if optimal_epochs is small
    # If optimal_epochs <= Config.SWA_START_EPOCH, we effectively just train standard models
    # But usually we expect optimal to be around 30-40.
    # We pass the calculated optimal_epochs as total_epochs.
    # We keep SWA_START_EPOCH from config, but if optimal < start, SWA won't trigger (handled in engine).

    for i in range(n_models):
        log_message(f"\nTraining Ensemble Model {i+1}/{n_models}")
        model = IcebergResNet18()

        save_path_swa = os.path.join(Config.CHECKPOINT_DIR, f"ensemble_{i}_swa.pth")

        run_swa_training(
            model,
            dl_full_train,
            dl_holdout_val,
            device=Config.DEVICE,
            total_epochs=optimal_epochs,
            save_path_best=None,  # Don't need best model, we want the final SWA/Standard state
            save_path_swa=save_path_swa,
            early_stopping=False,
        )

        # Load the saved SWA model (or the final state if SWA didn't run)
        if os.path.exists(save_path_swa):
            model = load_model(model, save_path_swa, device=Config.DEVICE)
            log_message("Loaded SWA weights.")
        else:
            # If SWA didn't run, the model in memory is the final epoch model
            log_message(
                "SWA did not run (optimal epochs < start). Using final model state."
            )

        ensemble_models.append(model)

    # 4. Final Evaluation on Hold-Out Validation Set
    log_message("\n=== Final Validation on Hold-Out Set ===")

    all_preds = []
    all_labels = []
    all_angles = []
    all_img_means = []

    for m in ensemble_models:
        m.eval()
        m.to(Config.DEVICE)

    with torch.no_grad():
        for images, angles, labels in dl_holdout_val:
            images = images.to(Config.DEVICE)
            angles_gpu = angles.to(Config.DEVICE)

            # Manual Ensemble TTA Prediction
            batch_preds = []
            images_h = torch.flip(images, dims=[3])
            images_v = torch.flip(images, dims=[2])

            for model in ensemble_models:
                l_orig = model(images, angles_gpu)
                l_h = model(images_h, angles_gpu)
                l_v = model(images_v, angles_gpu)
                p_avg = (
                    torch.sigmoid(l_orig) + torch.sigmoid(l_h) + torch.sigmoid(l_v)
                ) / 3.0
                batch_preds.append(p_avg)

            batch_preds = torch.stack(batch_preds).mean(dim=0)

            all_preds.extend(batch_preds.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())
            all_angles.extend(angles.numpy().flatten())
            img_means = images[:, 0, :, :].mean(dim=(1, 2)).cpu().numpy()
            all_img_means.extend(img_means)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_preds = np.clip(all_preds, 1e-15, 1 - 1e-15)

    final_metric = log_loss(all_labels, all_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    log_message("\n=== Failure Analysis ===")
    errors = np.abs(all_labels - all_preds)
    corr_angle, _ = pearsonr(errors, all_angles)
    print(f"Correlation (Error vs Inc Angle): {corr_angle:.4f}")
    corr_signal, _ = pearsonr(errors, all_img_means)
    print(f"Correlation (Error vs Signal Strength): {corr_signal:.4f}")

    # 6. Submission
    threshold = 0.16918645240183008
    if final_metric < threshold:
        log_message(
            f"\nMetric ({final_metric}) is better than threshold ({threshold}). Generating submission..."
        )
        from library.dataset import get_test_loader

        test_loader = get_test_loader(load_cached_data=True)
        test_predictions = predict_ensemble(
            ensemble_models, test_loader, device=Config.DEVICE
        )
        create_submission(test_predictions)
    else:
        log_message(
            f"\nMetric ({final_metric}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
