import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Import library modules
from library import config, utils, data, model, train


def run_pipeline():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override config for fast baseline execution
    config.NUM_EPOCHS = 50

    print(f"Configuration:")
    print(f"  Device: {config.DEVICE}")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Folds: {config.NUM_FOLDS}")

    utils.seed_everything(config.SEED)

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading data...")
    # Load all processed data (using cache if available)
    data_dict = utils.load_and_process_data(load_cached_data=True)

    # ==========================================
    # 3. Training Loop (5-Fold CV)
    # ==========================================
    print("\nStarting 5-Fold Cross-Validation...")
    fold_val_losses = []

    for fold_idx in range(config.NUM_FOLDS):
        val_loss = train.run_fold(fold_idx)
        fold_val_losses.append(val_loss)
        print(f"Fold {fold_idx} Best Val Loss: {val_loss:.6f}")

    print(f"\nAverage Fold Validation Loss: {np.mean(fold_val_losses):.6f}")

    # ==========================================
    # 4. Evaluation on Hold-out Validation Set
    # ==========================================
    print("\nEvaluating on Hold-out Validation Set (metadata/val.csv)...")

    # Load validation metadata to identify the fixed validation split
    val_meta_df = pd.read_csv(config.VAL_META_FILE)
    val_ids = set(val_meta_df["id"].values)

    # Filter the full training data to extract only the validation samples
    train_ids = data_dict["train_ids"]
    mask = np.isin(train_ids, list(val_ids))

    val_images = data_dict["train_images"][mask]
    val_targets = data_dict["train_targets"][mask]
    val_angles = data_dict["train_inc_angles"][mask]

    # Create DataLoader for the hold-out validation set
    val_dataset = data.IcebergDataset(
        images=val_images,
        inc_angles=val_angles,
        targets=val_targets,
        transform=data.get_transforms(mode="val"),
        stats=data_dict["stats"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Run Ensemble Prediction on Validation Set
    ensemble_preds = predict_ensemble(val_loader, config.NUM_FOLDS, has_targets=True)

    # Calculate and Print Metric
    final_metric = log_loss(val_targets, ensemble_preds)
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("\nPerforming Failure Analysis...")
    perform_failure_analysis(val_images, val_angles, val_targets, ensemble_preds)

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    THRESHOLD = 0.16398690884254846

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(data_dict)
    else:
        print(
            f"\nMetric ({final_metric}) did not beat threshold ({THRESHOLD}). Skipping submission generation."
        )


def predict_ensemble(loader, num_folds, has_targets=True):
    """
    Runs inference using an ensemble of trained models from all folds.
    Returns flattened probabilities.
    """
    device = config.DEVICE
    models = []

    # Load all trained models
    for i in range(num_folds):
        net = model.RDP_WBN().to(device)
        path = os.path.join(config.MODEL_CHECKPOINT_DIR, f"model_fold_{i}.pth")
        net.load_state_dict(torch.load(path, map_location=device))
        net.eval()
        models.append(net)

    all_preds = []

    with torch.no_grad():
        for batch in loader:
            # Unpack batch depending on whether targets are present
            if has_targets:
                images, angles, _ = batch
            else:
                images, angles = batch

            images = images.to(device)
            angles = angles.to(device)

            batch_preds = []
            # Get predictions from each model
            for net in models:
                logits = net(images, angles)
                probs = torch.sigmoid(logits)
                batch_preds.append(probs.cpu().numpy())

            # Average predictions across models (Ensemble)
            avg_preds = np.mean(batch_preds, axis=0)
            all_preds.append(avg_preds)

    return np.concatenate(all_preds).flatten()


def perform_failure_analysis(images, angles, targets, preds):
    """
    Analyzes the correlation between prediction error and input features.
    """
    # Calculate absolute errors
    errors = np.abs(targets - preds)

    # 1. Correlation with Incidence Angle
    valid_angle_mask = ~np.isnan(angles)
    if np.sum(valid_angle_mask) > 1:
        corr_angle, _ = pearsonr(errors[valid_angle_mask], angles[valid_angle_mask])
        print(f"Correlation between Error and Incidence Angle: {corr_angle:.4f}")
    else:
        print("Not enough valid incidence angles for correlation.")

    # 2. Correlation with Image Brightness (HH Band Mean)
    # images is (N, 3, 75, 75). Channel 0 is HH.
    hh_means = np.mean(images[:, 0, :, :], axis=(1, 2))
    corr_bright, _ = pearsonr(errors, hh_means)
    print(f"Correlation between Error and HH Band Brightness: {corr_bright:.4f}")

    # 3. Correlation with Image Contrast (HH Band Std)
    hh_stds = np.std(images[:, 0, :, :], axis=(1, 2))
    corr_contrast, _ = pearsonr(errors, hh_stds)
    print(f"Correlation between Error and HH Band Contrast: {corr_contrast:.4f}")


def generate_submission(data_dict):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating predictions for test set...")

    # Prepare Test Loader
    test_loader = data.get_test_dataloader(load_cached_data=True)

    # Run Ensemble Prediction
    flat_preds = predict_ensemble(test_loader, config.NUM_FOLDS, has_targets=False)

    # Create Submission DataFrame
    test_ids = data_dict["test_ids"]

    # Validation check
    if len(test_ids) != len(flat_preds):
        print(
            f"Warning: ID count ({len(test_ids)}) != Prediction count ({len(flat_preds)})"
        )

    submission = pd.DataFrame({"id": test_ids, "is_iceberg": flat_preds})

    # Save to CSV
    save_path = config.SUBMISSION_FILE
    submission.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


if __name__ == "__main__":
    run_pipeline()
