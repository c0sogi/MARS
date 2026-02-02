import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.model_selection import StratifiedKFold

# Import library modules
from library.config import Config
from library.dataset import load_data, IcebergDataset, get_transforms
from library.networks import IcebergResNet
from library.engine import set_seed, train_one_epoch, evaluate, update_bn_custom


def run_phase1(images, angles, labels, device):
    """
    Phase 1: Calibration using 5-Fold Cross-Validation (Cite 00040, 00049).
    Determines optimal epochs via Global Epoch Selection.
    """
    print(f"\n=== Phase 1: Calibration (5-Fold CV) ===")

    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Store validation losses: [Fold, Epoch]
    val_loss_history = np.zeros((Config.N_FOLDS, Config.PHASE1_MAX_EPOCHS))

    # To return for failure analysis (from the last fold)
    last_model = None
    last_val_loader = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
        print(f"\n--- Fold {fold + 1}/{Config.N_FOLDS} ---")

        # Prepare Data
        train_ds = IcebergDataset(
            images[train_idx],
            angles[train_idx],
            labels[train_idx],
            transform=get_transforms("train"),
        )
        val_ds = IcebergDataset(
            images[val_idx],
            angles[val_idx],
            labels[val_idx],
            transform=get_transforms("val"),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Model Setup
        model = IcebergResNet(
            pretrained=Config.PRETRAINED, dropout_rate=Config.DROPOUT_RATE
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            min_lr=Config.MIN_LR,
        )

        # Training Loop
        for epoch in range(1, Config.PHASE1_MAX_EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, epoch
            )
            # Use TTA=False for speed during calibration, but could use True for precision
            val_loss, val_acc = evaluate(
                model, val_loader, criterion, device, tta=False
            )

            scheduler.step(val_loss)
            val_loss_history[fold, epoch - 1] = val_loss

        last_model = model
        last_val_loader = val_loader

    # Global Epoch Selection
    # Calculate average validation loss across folds for each epoch
    avg_val_losses = np.mean(val_loss_history, axis=0)

    # Find the epoch with minimum average validation loss
    # +1 because epochs are 1-indexed
    optimal_epoch = np.argmin(avg_val_losses) + 1
    best_avg_val_loss = np.min(avg_val_losses)

    print(f"\nPhase 1 Complete.")
    print(f"Global Optimal Epoch: {optimal_epoch}")
    print(f"Best Averaged Val Loss: {best_avg_val_loss:.6f}")

    return last_model, best_avg_val_loss, optimal_epoch, last_val_loader


def analyze_failures(model, val_loader, device):
    """
    Perform failure analysis on the validation set.
    """
    print("\n=== Failure Analysis ===")
    model.eval()

    all_preds = []
    all_labels = []
    all_angles = []
    all_b1_means = []
    all_b2_means = []

    with torch.no_grad():
        for images, angles, labels in val_loader:
            images = images.to(device)
            angles_gpu = angles.to(device)

            # Predict
            logits = model(images, angles_gpu)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_preds.extend(probs)
            all_labels.extend(labels.numpy())
            all_angles.extend(angles.numpy())

            # Extract image stats (images are normalized, but relative intensity holds)
            # images shape: (B, 3, 224, 224). Channel 0 = Band 1, Channel 1 = Band 2
            imgs_np = images.cpu().numpy()
            all_b1_means.extend(np.mean(imgs_np[:, 0, :, :], axis=(1, 2)))
            all_b2_means.extend(np.mean(imgs_np[:, 1, :, :], axis=(1, 2)))

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate Error
    errors = np.abs(all_preds - all_labels)

    # Create DataFrame for correlation
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": all_angles,
            "b1_mean": all_b1_means,
            "b2_mean": all_b2_means,
        }
    )

    # Compute correlations
    correlations = df_analysis.corrwith(df_analysis["error"])
    print("Correlation between Error Magnitude and Features:")
    print(correlations.drop("error"))

    return df_analysis


def run_phase2(full_train_loader, optimal_epoch, device, num_models=3):
    """
    Phase 2: Production. Train SWA models on full dataset.
    """
    print(f"\n=== Phase 2: Production (Full-Fit SWA) | Ensemble Size: {num_models} ===")

    swa_models = []
    swa_epochs = 6  # Reduced SWA epochs for speed

    for i in range(num_models):
        print(f"Training Model {i+1}/{num_models}...")

        # Initialize fresh model
        model = IcebergResNet(
            pretrained=Config.PRETRAINED, dropout_rate=Config.DROPOUT_RATE
        ).to(device)
        swa_model = AveragedModel(model).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Standard Training Phase
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
        )

        # Train for optimal_epoch
        for epoch in range(1, optimal_epoch + 1):
            # We don't have a val set here, so we step scheduler based on train loss (proxy)
            train_loss, _ = train_one_epoch(
                model, full_train_loader, optimizer, criterion, device, epoch
            )
            scheduler.step(train_loss)

        # SWA Phase
        swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR)

        print(f"Starting SWA phase for {swa_epochs} epochs...")
        for epoch in range(optimal_epoch + 1, optimal_epoch + swa_epochs + 1):
            train_one_epoch(
                model, full_train_loader, optimizer, criterion, device, epoch
            )
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # Update BN statistics
        print("Updating BN statistics...")
        update_bn_custom(full_train_loader, swa_model, device)

        swa_models.append(swa_model)

    return swa_models


def generate_submission(models, test_loader, device):
    """
    Generate predictions using ensemble and TTA.
    """
    print("\n=== Generating Submission ===")

    # Prepare storage
    # We need to map predictions back to IDs.
    # The test_loader returns (images, angles, ids)

    results = {}  # id -> list of probs

    for model in models:
        model.eval()

    with torch.no_grad():
        for images, angles, ids in test_loader:
            images = images.to(device)
            angles = angles.to(device)

            batch_probs = np.zeros((images.size(0),), dtype=np.float32)

            # Ensemble averaging
            for model in models:
                # TTA: Original
                logits = model(images, angles)
                probs = torch.sigmoid(logits)

                # TTA: Horizontal Flip
                images_h = torch.flip(images, [3])
                logits_h = model(images_h, angles)
                probs_h = torch.sigmoid(logits_h)

                # TTA: Vertical Flip
                images_v = torch.flip(images, [2])
                logits_v = model(images_v, angles)
                probs_v = torch.sigmoid(logits_v)

                # Average TTA for this model
                model_avg = (probs + probs_h + probs_v) / 3.0
                batch_probs += model_avg.cpu().numpy().flatten()

            # Average over ensemble
            batch_probs /= len(models)

            # Store
            for img_id, prob in zip(ids, batch_probs):
                results[img_id] = prob

    # Create DataFrame
    submission_df = pd.DataFrame(list(results.items()), columns=["id", "is_iceberg"])

    # Save
    submission_path = Config.SUBMISSION_PATH
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    # Load raw data arrays (cached)
    train_imgs, train_angs, train_lbls, train_ids = load_data(
        Config.TRAIN_META_PATH, Config.TRAIN_JSON, "train"
    )
    val_imgs, val_angs, val_lbls, val_ids = load_data(
        Config.VAL_META_PATH, Config.TRAIN_JSON, "val"
    )
    test_imgs, test_angs, _, test_ids = load_data(
        Config.TEST_META_PATH, Config.TEST_JSON, "test"
    )

    # Combine Train and Val for 5-Fold CV (Phase 1)
    all_imgs = np.concatenate([train_imgs, val_imgs])
    all_angs = np.concatenate([train_angs, val_angs])
    all_lbls = np.concatenate([train_lbls, val_lbls])

    # Test Dataset
    test_dataset = IcebergDataset(
        test_imgs, test_angs, ids=test_ids, transform=get_transforms("test")
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Phase 1: Calibration (5-Fold CV)
    # We pass the full labeled dataset to run_phase1
    analysis_model, val_metric, optimal_epoch, analysis_loader = run_phase1(
        all_imgs, all_angs, all_lbls, device
    )

    # REQUIRED OUTPUT: Final Validation Metric
    print(f"Final Validation Metric: {val_metric}")

    # 4. Failure Analysis (using the last fold model)
    analyze_failures(analysis_model, analysis_loader, device)

    # 5. Conditional Submission
    THRESHOLD = 0.16918645240183008

    if val_metric < THRESHOLD:
        print(
            f"\nValidation metric ({val_metric:.6f}) meets threshold ({THRESHOLD:.6f}). Proceeding to submission."
        )

        # Prepare Full Dataset Loader for Phase 2
        full_dataset = IcebergDataset(
            all_imgs, all_angs, all_lbls, transform=get_transforms("train")
        )
        full_loader = DataLoader(
            full_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Run Phase 2
        swa_models = run_phase2(full_loader, optimal_epoch, device, num_models=5)

        # Generate Submission
        generate_submission(swa_models, test_loader, device)

    else:
        print(
            f"\nValidation metric ({val_metric:.6f}) did NOT meet threshold ({THRESHOLD:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
