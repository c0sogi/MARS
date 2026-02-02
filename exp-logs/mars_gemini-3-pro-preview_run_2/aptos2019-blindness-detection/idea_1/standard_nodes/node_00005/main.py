import os
import sys
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, quadratic_weighted_kappa
from library.dataset import load_data, RetinopathyDataset, get_transforms
from library.model import RetinopathyRegressor
from library.engine import run_training


def get_predictions(model, loader, device, is_test=False):
    """
    Runs inference on a dataloader and returns predictions (and targets if not test).
    """
    model.eval()
    preds = []
    targets_or_ids = []

    with torch.no_grad():
        for batch in loader:
            if is_test:
                images, ids = batch
                images = images.to(device, dtype=torch.float)
                outputs = model(images)
                preds.extend(outputs.cpu().numpy().tolist())
                targets_or_ids.extend(ids)
            else:
                images, targets = batch
                images = images.to(device, dtype=torch.float)
                outputs = model(images)
                preds.extend(outputs.cpu().numpy().tolist())
                targets_or_ids.extend(targets.numpy().tolist())

    return np.array(preds), np.array(targets_or_ids)


def perform_failure_analysis(df, preds, targets):
    """
    Analyzes correlation between error magnitude and image metadata.
    """
    print("\n=== Failure Analysis ===")

    # Calculate absolute error
    # Preds are continuous here, targets are integers
    # We analyze error on the continuous prediction to see sensitivity
    error = np.abs(preds - targets)

    # Collect image stats
    stats = []
    print("Extracting metadata from validation images for analysis...")
    for idx, row in df.iterrows():
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        try:
            # File size
            size = os.path.getsize(file_path)

            # Image dimensions (read header only if possible, but cv2 reads full)
            img = cv2.imread(file_path)
            if img is not None:
                h, w = img.shape[:2]
                aspect_ratio = w / h if h > 0 else 0
            else:
                h, w, aspect_ratio = 0, 0, 0

            stats.append(
                {
                    "width": w,
                    "height": h,
                    "aspect_ratio": aspect_ratio,
                    "file_size": size,
                }
            )
        except Exception:
            stats.append({"width": 0, "height": 0, "aspect_ratio": 0, "file_size": 0})

    stats_df = pd.DataFrame(stats)
    stats_df["error"] = error

    # Compute correlations
    # We use Spearman because relationships might not be strictly linear
    correlations = stats_df.corr(method="spearman")["error"].drop("error")

    print("Correlation between Model Error and Input Features:")
    print(correlations)

    return correlations


def main():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    print(f"Device: {Config.DEVICE}")

    # 2. Data Loading
    print("Loading data...")
    train_df, val_df, test_df = load_data(load_cached_data=True)

    # Datasets
    train_ds = RetinopathyDataset(
        train_df, transforms=get_transforms("train"), mode="train"
    )
    val_ds = RetinopathyDataset(
        val_df, transforms=get_transforms("valid"), mode="valid"
    )
    test_ds = RetinopathyDataset(
        test_df, transforms=get_transforms("test"), mode="test"
    )

    # DataLoaders
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

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = RetinopathyRegressor(pretrained=True)
    model = model.to(Config.DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Training
    print("Starting training...")

    # Initialize Scheduler (Cite solution_lesson_node_00003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6
    )

    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        num_epochs=Config.NUM_EPOCHS,
        patience=8,  # Relaxed patience (Cite solution_lesson_node_00003)
    )

    # 5. Validation & Metrics
    print("\nLoading best model for validation...")
    model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    )
    model.eval()

    # Get raw predictions (continuous)
    val_raw_preds, val_targets = get_predictions(
        model, val_loader, Config.DEVICE, is_test=False
    )

    # Post-process for metric calculation
    val_preds_clipped = np.clip(val_raw_preds, 0, 4)
    val_preds_rounded = np.round(val_preds_clipped).astype(int)
    val_targets_int = val_targets.astype(int)

    # Compute Metric
    final_metric = quadratic_weighted_kappa(val_targets_int, val_preds_rounded)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    perform_failure_analysis(val_df, val_raw_preds, val_targets)

    # 7. Test Inference & Submission
    if final_metric > 0.889313457904682:
        print("\nGenerating submission...")
        test_raw_preds, test_ids = get_predictions(
            model, test_loader, Config.DEVICE, is_test=True
        )

        # Post-process
        test_preds_clipped = np.clip(test_raw_preds, 0, 4)
        test_preds_rounded = np.round(test_preds_clipped).astype(int)

        # Create submission DataFrame
        submission_df = pd.DataFrame(
            {"id_code": test_ids, "diagnosis": test_preds_rounded}
        )

        # Save
        submission_df.to_csv(Config.SUBMISSION_SAVE_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_SAVE_PATH}")
        print(submission_df.head())
    else:
        print(
            f"\nValidation metric {final_metric:.4f} did not meet threshold 0.8882. Submission skipped."
        )


if __name__ == "__main__":
    main()
