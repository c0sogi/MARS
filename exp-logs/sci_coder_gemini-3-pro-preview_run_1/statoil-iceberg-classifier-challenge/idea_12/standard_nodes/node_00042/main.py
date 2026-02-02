import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings

# Import provided library functions and classes
from library.config import Config
from library.utils import seed_everything, calculate_log_loss
from library.data import process_and_cache_data, IcebergDataset, get_transforms
from library.model import IcebergResNet18
from library.engine import train_one_epoch, evaluate

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_training(data, train_indices, val_indices, device):
    """
    Executes Single-Split Training with TTA-Aware Checkpointing.
    Cite solution_lesson_node_00020 (Maximize Data), solution_lesson_node_00025 (TTA Gap).
    """
    print("\n=== Single-Split Training with TTA-Aware Checkpointing ===")

    # Unpack data arrays
    images = data["train_images"]
    angles = data["train_angles"]
    labels = data["train_labels"]
    stats = data["stats"]

    # Create Datasets using Metadata Splits
    train_ds = IcebergDataset(
        images[train_indices],
        angles[train_indices],
        labels[train_indices],
        transform=get_transforms("train"),
        angle_stats=stats,
    )
    val_ds = IcebergDataset(
        images[val_indices],
        angles[val_indices],
        labels[val_indices],
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
    # Reactive Scheduler (Cite solution_lesson_node_00038)
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
    es_counter = 0

    import copy

    for epoch in range(1, Config.MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Evaluate with TTA for Checkpointing (Cite solution_lesson_node_00025)
        # We use TTA loss to drive Early Stopping and Checkpointing
        val_loss, val_metric, _, _ = evaluate(model, val_loader, device, use_tta=True)

        print(
            f"Validation Loss (TTA): {val_loss:.4f} | Log Loss (TTA): {val_metric:.4f}"
        )

        # Scheduler steps on Validation Loss
        scheduler.step(val_loss)

        # Checkpoint based on TTA Loss
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            # Deep Copy to avoid reference bugs (Cite solution_lesson_node_00002)
            best_model_state = copy.deepcopy(model.state_dict())
            es_counter = 0
        else:
            es_counter += 1

        if es_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Best Epoch: {best_epoch}, Best Val Loss (TTA): {best_loss:.4f}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def perform_inference_tta(model, loader, device):
    """
    Performs Test Time Augmentation (Original, HFlip, VFlip).
    Returns IDs (or labels) and probabilities.
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

            # TTA Variations: Original, Horizontal Flip, Vertical Flip (Cite solution_lesson_node_00031)
            variations = [
                images,
                torch.flip(images, [3]),  # HFlip
                torch.flip(images, [2]),  # VFlip
            ]

            logits_sum = None

            model.eval()
            for img_var in variations:
                logits = model(img_var, angles)
                if logits_sum is None:
                    logits_sum = logits
                else:
                    logits_sum += logits

            # Average Logits
            logits_avg = logits_sum / len(variations)
            probs = torch.sigmoid(logits_avg)

            avg_probs.append(probs.cpu().numpy().flatten())

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

    # 2. Train Single Model (Cite solution_lesson_node_00020)
    model = run_training(data, train_indices, val_indices, device)

    # 3. Final Validation (using the trained model)
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

    val_labels_extracted, val_preds = perform_inference_tta(model, val_loader, device)

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

        test_ids, test_probs = perform_inference_tta(model, test_loader, device)

        df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": test_probs})

        os.makedirs("./submission", exist_ok=True)
        submission_path = "./submission/submission.csv"
        df_sub.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(f"\nMetric {final_metric:.6f} >= {threshold}. Skipping submission.")


if __name__ == "__main__":
    main()
