import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import warnings

# Import provided library functions and classes
from library.config import Config
from library.utils import seed_everything, calculate_log_loss
from library.data import process_and_cache_data, IcebergDataset, get_transforms
from library.model import IcebergResNet18
from library.engine import train_one_epoch, evaluate
from library.calibration import PlattScaler

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_phase_1_cv(data, train_indices, device):
    """
    Executes Phase 1: Stratified 5-Fold CV to find optimal epoch, scheduler milestones,
    and fit the Platt Scaler on OOF logits.
    """
    print("\n=== Phase 1: Calibration & Trajectory Extraction (5-Fold CV) ===")

    # Unpack data arrays
    images = data["train_images"]
    angles = data["train_angles"]
    labels = data["train_labels"]
    stats = data["stats"]

    # Subset to the training metadata indices
    X = images[train_indices]
    a = angles[train_indices]
    y = labels[train_indices]

    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    fold_best_epochs = []
    fold_milestones = []

    # Storage for OOF logits and labels for calibration
    all_oof_logits = []
    all_oof_labels = []

    for fold, (t_idx, v_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{Config.NUM_FOLDS} ---")

        # Create Datasets
        train_ds = IcebergDataset(
            X[t_idx],
            a[t_idx],
            y[t_idx],
            transform=get_transforms("train"),
            angle_stats=stats,
        )
        val_ds = IcebergDataset(
            X[v_idx],
            a[v_idx],
            y[v_idx],
            transform=get_transforms("val"),
            angle_stats=stats,
        )

        # Create Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model, Optimizer, Scheduler
        model = IcebergResNet18().to(device)
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

        best_loss = float("inf")
        best_epoch = 0
        milestones = []
        current_lr = Config.LEARNING_RATE

        # Store best logits for this fold
        best_fold_logits = None
        best_fold_labels = None

        es_counter = 0

        for epoch in range(1, Config.MAX_EPOCHS + 1):
            _ = train_one_epoch(model, train_loader, optimizer, device, epoch)
            val_loss, _, logits, targets = evaluate(model, val_loader, device)

            # Scheduler Step & Milestone Tracking
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < current_lr:
                milestones.append(epoch)
                current_lr = new_lr

            # Checkpoint
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_fold_logits = logits
                best_fold_labels = targets
                es_counter = 0
            else:
                es_counter += 1

            if es_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

        fold_best_epochs.append(best_epoch)
        fold_milestones.append(milestones)
        all_oof_logits.append(best_fold_logits)
        all_oof_labels.append(best_fold_labels)

        print(f"Fold {fold+1} Best Epoch: {best_epoch}, Best Val Loss: {best_loss:.4f}")

    # Calculate Global Optimal Epoch
    avg_best_epoch = int(np.mean(fold_best_epochs))

    # Extract Average Milestones
    max_drops = max([len(m) for m in fold_milestones]) if fold_milestones else 0
    avg_milestones = []
    if max_drops > 0:
        for i in range(max_drops):
            drops = [m[i] for m in fold_milestones if len(m) > i]
            if drops:
                avg_milestones.append(int(np.mean(drops)))

    print(f"\nPhase 1 Complete.")
    print(f"Optimal Epoch (E*): {avg_best_epoch}")
    print(f"Trajectory Milestones: {avg_milestones}")

    # Fit Calibration Model
    print("Fitting Platt Scaler on OOF data...")
    scaler = PlattScaler()
    concatenated_logits = np.concatenate(all_oof_logits)
    concatenated_labels = np.concatenate(all_oof_labels)
    scaler.fit(concatenated_logits, concatenated_labels)

    return avg_best_epoch, avg_milestones, scaler


def run_phase_2_full_fit(data, train_indices, device, optimal_epoch, milestones):
    """
    Executes Phase 2: Train 5 models on the full training set using the extracted trajectory.
    """
    print("\n=== Phase 2: Full-Fit Ensemble Training (Trajectory Replay) ===")

    images = data["train_images"]
    angles = data["train_angles"]
    labels = data["train_labels"]
    stats = data["stats"]

    # Full Train Set
    X = images[train_indices]
    a = angles[train_indices]
    y = labels[train_indices]

    models_list = []

    # Train 5 independent models
    for i in range(5):
        print(f"\n--- Training Full-Fit Model {i + 1}/5 ---")

        ds = IcebergDataset(
            X, a, y, transform=get_transforms("train"), angle_stats=stats
        )
        loader = DataLoader(
            ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        model = IcebergResNet18().to(device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Trajectory Replay Scheduler
        if milestones:
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=Config.SCHEDULER_FACTOR
            )
        else:
            scheduler = None

        for epoch in range(1, optimal_epoch + 1):
            _ = train_one_epoch(model, loader, optimizer, device, epoch)
            if scheduler:
                scheduler.step()

        models_list.append(model)

    return models_list


def perform_inference_tta(models_list, loader, device, scaler):
    """
    Performs Test Time Augmentation (Original, HFlip, VFlip) and Calibration.
    Returns IDs (or labels) and calibrated probabilities.
    """
    print("Starting TTA Inference...")

    avg_probs = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            # Unpack batch (handles both val (img, ang, lbl) and test (img, ang, id))
            if len(batch) == 3:
                images, angles, extras = batch
            else:
                raise ValueError("Unexpected batch structure")

            images = images.to(device)
            angles = angles.to(device)

            batch_probs_sum = None

            # TTA Variations: Original, Horizontal Flip, Vertical Flip
            variations = [
                images,
                torch.flip(images, [3]),  # HFlip
                torch.flip(images, [2]),  # VFlip
            ]

            for img_var in variations:
                for model in models_list:
                    model.eval()
                    logits = model(img_var, angles)
                    probs = torch.sigmoid(logits)

                    if batch_probs_sum is None:
                        batch_probs_sum = probs
                    else:
                        batch_probs_sum += probs

            # Average over (3 variations * 5 models)
            batch_avg_probs = batch_probs_sum / (len(variations) * len(models_list))

            # Apply Calibration
            batch_avg_probs_np = batch_avg_probs.cpu().numpy().flatten()
            batch_cal_probs = scaler.predict_from_probs(batch_avg_probs_np)

            avg_probs.append(batch_cal_probs)

            # Collect IDs or Labels
            if isinstance(extras, torch.Tensor):
                all_ids.extend(extras.cpu().numpy().flatten())
            else:
                all_ids.extend(extras)

    final_probs = np.concatenate(avg_probs)
    final_ids = np.array(all_ids)

    return final_ids, final_probs


def failure_analysis(val_preds, val_labels, data, val_indices):
    """
    Analyzes failure modes on the validation set.
    """
    print("\n=== Failure Analysis ===")

    errors = np.abs(val_labels - val_preds)
    log_losses = [calculate_log_loss([y], [p]) for y, p in zip(val_labels, val_preds)]

    # Retrieve incidence angles for the validation set
    angles = data["train_angles"][val_indices]

    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "log_loss": log_losses,
            "inc_angle": angles,
            "label": val_labels,
            "pred": val_preds,
        }
    )

    # Correlation Analysis
    corr = df_analysis["error"].corr(df_analysis["inc_angle"])
    print(f"Correlation between Error and Incidence Angle: {corr:.10f}")

    print("Top 5 High Error Samples:")
    print(
        df_analysis.sort_values("log_loss", ascending=False).head(5)[
            ["inc_angle", "label", "pred", "log_loss"]
        ]
    )


def main():
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Load Data & Metadata
    df_train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    df_val_meta = pd.read_csv(Config.VAL_META_PATH)
    df_test_meta = pd.read_csv(Config.TEST_META_PATH)

    train_indices = df_train_meta["sample_index"].values
    val_indices = df_val_meta["sample_index"].values
    test_indices = df_test_meta["sample_index"].values

    data = process_and_cache_data(load_cached_data=True)

    # 2. Phase 1: CV & Calibration
    optimal_epoch, milestones, scaler = run_phase_1_cv(data, train_indices, device)

    # 3. Phase 2: Full-Fit Ensemble
    models_list = run_phase_2_full_fit(
        data, train_indices, device, optimal_epoch, milestones
    )

    # 4. Final Validation
    print("\n=== Final Validation ===")
    val_ds = IcebergDataset(
        data["train_images"][val_indices],
        data["train_angles"][val_indices],
        data["train_labels"][val_indices],
        transform=get_transforms("val"),
        angle_stats=data["stats"],
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_labels_extracted, val_preds = perform_inference_tta(
        models_list, val_loader, device, scaler
    )

    # Metric
    final_metric = calculate_log_loss(val_labels_extracted, val_preds)
    print(f"Final Validation Metric: {final_metric:.15f}")

    # Failure Analysis
    failure_analysis(val_preds, val_labels_extracted, data, val_indices)

    # 5. Submission
    threshold = 0.17822679498532543
    if final_metric < threshold:
        print(f"\nMetric {final_metric:.6f} < {threshold}. Generating submission...")

        test_ds = IcebergDataset(
            data["test_images"][test_indices],
            data["test_angles"][test_indices],
            ids=data["test_ids"][test_indices],
            transform=get_transforms("test"),
            angle_stats=data["stats"],
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_ids, test_probs = perform_inference_tta(
            models_list, test_loader, device, scaler
        )

        df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": test_probs})

        os.makedirs("./submission", exist_ok=True)
        submission_path = "./submission/submission.csv"
        df_sub.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(f"\nMetric {final_metric:.6f} >= {threshold}. Skipping submission.")


if __name__ == "__main__":
    main()
