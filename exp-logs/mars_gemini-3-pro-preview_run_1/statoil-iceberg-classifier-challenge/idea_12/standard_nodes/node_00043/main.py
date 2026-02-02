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


import copy


def train_seed_ensemble(data, train_indices, val_indices, device, num_seeds=5):
    """
    Trains an ensemble of models using different random seeds on the same 80/20 split.
    Uses Dynamic Early Stopping for each model.
    Cite solution_lesson_node_00020: Maximize training set size (80% vs 64% in CV).
    """
    print(f"\n=== Training Seed Ensemble ({num_seeds} models) ===")

    # Unpack data
    images = data["train_images"]
    angles = data["train_angles"]
    labels = data["train_labels"]
    stats = data["stats"]

    # Fixed Split
    X_train, a_train, y_train = (
        images[train_indices],
        angles[train_indices],
        labels[train_indices],
    )
    X_val, a_val, y_val = images[val_indices], angles[val_indices], labels[val_indices]

    models_list = []

    for i in range(num_seeds):
        seed = Config.SEED + i
        seed_everything(seed)
        print(f"\n--- Model {i+1}/{num_seeds} (Seed {seed}) ---")

        # Create Datasets
        train_ds = IcebergDataset(
            X_train,
            a_train,
            y_train,
            transform=get_transforms("train"),
            angle_stats=stats,
        )
        val_ds = IcebergDataset(
            X_val, a_val, y_val, transform=get_transforms("val"), angle_stats=stats
        )

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

        # Model & Optimizer
        model = IcebergResNet18().to(device)
        # Cite solution_lesson_node_00007: AdamW needs higher weight decay (1e-2)
        optimizer = optim.AdamW(
            model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-2
        )

        # Cite solution_lesson_node_00038: Use Reactive Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            min_lr=Config.MIN_LR,
        )

        best_loss = float("inf")
        best_model_state = None
        es_counter = 0

        for epoch in range(1, Config.MAX_EPOCHS + 1):
            _ = train_one_epoch(model, train_loader, optimizer, device, epoch)
            val_loss, _, _, _ = evaluate(model, val_loader, device)

            scheduler.step(val_loss)

            # Cite solution_lesson_node_00002: Deep copy state dict
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                es_counter = 0
            else:
                es_counter += 1

            if es_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

        # Restore best model
        if best_model_state:
            model.load_state_dict(best_model_state)
            print(f"Restored best model with Val Loss: {best_loss:.4f}")

        models_list.append(model)

    return models_list


def perform_inference_tta(models_list, loader, device):
    """
    Performs Test Time Augmentation (Original, HFlip, VFlip).
    Returns IDs (or labels) and averaged probabilities.
    """
    print("Starting TTA Inference...")

    avg_probs = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                images, angles, extras = batch
            else:
                raise ValueError("Unexpected batch structure")

            images = images.to(device)
            angles = angles.to(device)

            batch_probs_sum = None

            # TTA Variations: Original, HFlip, VFlip
            variations = [
                images,
                torch.flip(images, [3]),
                torch.flip(images, [2]),
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

            # Average over (3 variations * N models)
            batch_avg_probs = batch_probs_sum / (len(variations) * len(models_list))

            avg_probs.append(batch_avg_probs.cpu().numpy().flatten())

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

    # 2. Train Seed Ensemble
    # Cite solution_lesson_node_00041: Avoid fixed-epoch replay; use dynamic early stopping.
    models_list = train_seed_ensemble(
        data, train_indices, val_indices, device, num_seeds=5
    )

    # 3. Final Validation
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
        models_list, val_loader, device
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

        test_ids, test_probs = perform_inference_tta(models_list, test_loader, device)

        df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": test_probs})

        os.makedirs("./submission", exist_ok=True)
        submission_path = "./submission/submission.csv"
        df_sub.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(f"\nMetric {final_metric:.6f} >= {threshold}. Skipping submission.")


if __name__ == "__main__":
    main()
