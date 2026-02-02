import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.swa_utils import AveragedModel, SWALR

# Import library modules
from library.config import Config
from library.dataset import load_data, IcebergDataset, get_transforms
from library.networks import IcebergResNet
from library.engine import set_seed, train_one_epoch, evaluate, update_bn_custom


def run_phase1(train_loader, val_loader, device):
    """
    Phase 1: Calibration. Train on train split, validate on val split.
    Determines optimal epochs and provides validation metric.
    """
    print("\n=== Phase 1: Calibration & Validation ===")

    model = IcebergResNet(
        pretrained=Config.PRETRAINED, dropout_rate=Config.DROPOUT_RATE
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.MIN_LR,
    )

    best_loss = float("inf")
    best_epoch = 0
    best_model_state = None

    # Limit epochs for fast baseline execution
    max_epochs = 20

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, tta=False)

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            # print(f"New Best Epoch: {epoch} | Val Loss: {val_loss:.6f}")

    print(f"Phase 1 Complete. Best Epoch: {best_epoch}, Best Val Loss: {best_loss:.6f}")

    # Load best model for analysis
    model.load_state_dict(best_model_state)

    return model, best_loss, best_epoch


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

    # Create Datasets
    train_dataset = IcebergDataset(
        train_imgs, train_angs, train_lbls, transform=get_transforms("train")
    )
    val_dataset = IcebergDataset(
        val_imgs, val_angs, val_lbls, transform=get_transforms("val")
    )
    test_dataset = IcebergDataset(
        test_imgs, test_angs, ids=test_ids, transform=get_transforms("test")
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Phase 1: Calibration & Validation
    best_model, val_metric, optimal_epoch = run_phase1(train_loader, val_loader, device)

    # REQUIRED OUTPUT: Final Validation Metric
    print(f"Final Validation Metric: {val_metric}")

    # 4. Failure Analysis
    analyze_failures(best_model, val_loader, device)

    # 5. Conditional Submission
    THRESHOLD = 0.16918645240183008

    if val_metric < THRESHOLD:
        print(
            f"\nValidation metric ({val_metric:.6f}) meets threshold ({THRESHOLD:.6f}). Proceeding to submission."
        )

        # Prepare Full Dataset (Train + Val) for Phase 2
        # We concatenate the underlying arrays
        full_imgs = np.concatenate([train_imgs, val_imgs])
        full_angs = np.concatenate([train_angs, val_angs])
        full_lbls = np.concatenate([train_lbls, val_lbls])

        full_dataset = IcebergDataset(
            full_imgs, full_angs, full_lbls, transform=get_transforms("train")
        )
        full_loader = DataLoader(
            full_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Run Phase 2
        swa_models = run_phase2(full_loader, optimal_epoch, device, num_models=3)

        # Generate Submission
        generate_submission(swa_models, test_loader, device)

    else:
        print(
            f"\nValidation metric ({val_metric:.6f}) did NOT meet threshold ({THRESHOLD:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
