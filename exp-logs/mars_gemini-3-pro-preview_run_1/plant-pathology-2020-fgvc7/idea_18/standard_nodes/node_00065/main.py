import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import cv2
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import (
    seed_everything,
    calculate_roc_auc,
    get_class_weights,
    check_initial_loss,
)
from library.data import get_loaders, get_test_loader, AppleDataset, get_transforms
from library.model import AppleResNet34
from library.engine import train_one_epoch, validate


def run_failure_analysis(val_df, y_true, y_pred_probs):
    """
    Analyzes correlations between error magnitude and image meta-features.
    """
    print("\n==== Failure Analysis ====")

    # Calculate Error Magnitude: 1 - probability assigned to the true class
    # y_true is (N, 4) (one-hot or probs), y_pred_probs is (N, 4)
    # We need to find the index of the true class for each sample
    true_indices = np.argmax(y_true, axis=1)

    # Extract the predicted probability for the true class
    prob_true_class = y_pred_probs[np.arange(len(y_true)), true_indices]
    error_magnitude = 1.0 - prob_true_class

    # Extract Meta-Features
    widths = []
    heights = []
    intensities = []

    for _, row in val_df.iterrows():
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                h, w, c = img.shape
                # Calculate mean intensity (normalized)
                mean_intensity = img.mean() / 255.0

                widths.append(w)
                heights.append(h)
                intensities.append(mean_intensity)
            else:
                widths.append(0)
                heights.append(0)
                intensities.append(0)
        else:
            widths.append(0)
            heights.append(0)
            intensities.append(0)

    # Calculate Correlations
    if len(widths) == len(error_magnitude):
        corr_w, _ = pearsonr(error_magnitude, widths)
        corr_h, _ = pearsonr(error_magnitude, heights)
        corr_i, _ = pearsonr(error_magnitude, intensities)

        print(f"Correlation between Error and Width: {corr_w:.4f}")
        print(f"Correlation between Error and Height: {corr_h:.4f}")
        print(f"Correlation between Error and Intensity: {corr_i:.4f}")
    else:
        print("Mismatch in data lengths, skipping correlation analysis.")


def main():
    # 1. Initialization
    seed_everything(Config.SEED)
    Config.setup()
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Setup & Class Weights
    # We need weights for the loss function. Use ONLY training data.
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

    # Disable cache to ensure weights are calculated on train set only (Cite debug_lesson_2)
    class_weights = get_class_weights(
        train_meta, target_cols, load_cached_data=False
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 3. Training Phase (Seed Averaging)
    # Cite solution_lesson_node_00055: Seed Averaging Ensembles for Robustness
    print("\n==== Starting Seed Averaging Training ====")

    for seed_idx in range(Config.NUM_SEEDS):
        current_seed = Config.SEED + seed_idx
        seed_everything(current_seed)
        print(
            f"\nTraining Seed {seed_idx + 1}/{Config.NUM_SEEDS} (Seed: {current_seed})"
        )

        # Get Loaders (Same data, different seed effects on shuffling/aug)
        train_loader, val_loader = get_loaders()

        # Initialize Model
        model = AppleResNet34(
            num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED
        )
        model.to(device)

        # Optimizer & Scheduler
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=Config.T_0, T_mult=Config.T_MULT, eta_min=Config.ETA_MIN
        )

        # Initial Loss Check (only for first seed)
        if seed_idx == 0:
            check_initial_loss(model, train_loader, criterion, device)

        # Training Loop
        best_auc = 0.0
        best_model_path = os.path.join(Config.MODELS_DIR, f"seed_{seed_idx}_best.pth")

        for epoch in range(Config.EPOCHS):
            train_loss, train_auc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_auc = validate(model, val_loader, criterion, device)

            scheduler.step()

            # Save best model based on Validation AUC
            # Cite solution_lesson_node_00031: Metric-Based Model Checkpointing
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_path)

        print(f"Seed {seed_idx + 1} Finished. Best Val AUC: {best_auc:.4f}")

    # 4. Validation Assessment (Ensemble on Fixed Hold-out Set)
    print("\n==== Final Validation Assessment ====")

    # Load fixed validation set
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    val_dataset = AppleDataset(val_df, transforms=get_transforms("valid"))
    val_loader_fixed = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load all models
    models = []
    for seed_idx in range(Config.NUM_SEEDS):
        model_path = os.path.join(Config.MODELS_DIR, f"seed_{seed_idx}_best.pth")
        if os.path.exists(model_path):
            m = AppleResNet34(num_classes=Config.NUM_CLASSES, pretrained=False)
            m.load_state_dict(torch.load(model_path, map_location=device))
            m.to(device)
            m.eval()
            models.append(m)

    if not models:
        print("No models trained. Exiting.")
        return

    # Ensemble Inference
    all_targets = []
    all_preds_avg = []

    with torch.no_grad():
        for images, targets in val_loader_fixed:
            images = images.to(device)

            # Get predictions from all models
            batch_preds = []
            for m in models:
                outputs = m(images)
                probs = torch.softmax(outputs, dim=1)
                batch_preds.append(probs.cpu().numpy())

            # Average predictions
            avg_preds = np.mean(batch_preds, axis=0)

            all_targets.append(targets.numpy())
            all_preds_avg.append(avg_preds)

    all_targets = np.concatenate(all_targets)
    all_preds_avg = np.concatenate(all_preds_avg)

    final_metric = calculate_roc_auc(all_targets, all_preds_avg)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    run_failure_analysis(val_df, all_targets, all_preds_avg)

    # 6. Submission
    threshold = 0.9901680711448418
    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric:.6f}) > Threshold ({threshold:.6f}). Generating submission..."
        )

        test_loader = get_test_loader()
        test_preds_list = []

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)

                batch_preds = []
                for m in models:
                    outputs = m(images)
                    probs = torch.softmax(outputs, dim=1)
                    batch_preds.append(probs.cpu().numpy())

                avg_preds = np.mean(batch_preds, axis=0)
                test_preds_list.append(avg_preds)

        test_preds_flat = np.concatenate(test_preds_list)

        # Create submission DataFrame
        test_df = pd.read_csv(Config.TEST_METADATA_PATH)
        image_ids = test_df["image_id"].values

        submission_df = pd.DataFrame(test_preds_flat, columns=target_cols)
        submission_df.insert(0, "image_id", image_ids)

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric ({final_metric:.6f}) <= Threshold ({threshold:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
