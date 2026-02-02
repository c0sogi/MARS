import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.utils import set_seed, calculate_multilabel_auc
from library.dataset import BirdDataset, get_train_transforms, get_valid_transforms
from library.models import BirdClassifier
from library.sam import SAM
from library.engine import (
    get_weighted_loss,
    train_one_epoch,
    evaluate,
    predict_cyclic_tta,
)


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    # Load datasets using the metadata CSVs
    train_dataset = BirdDataset(
        Config.TRAIN_CSV,
        mode="train",
        load_cached_data=True,
        transforms=get_train_transforms(),
    )
    val_dataset = BirdDataset(
        Config.VAL_CSV,
        mode="val",
        load_cached_data=True,
        transforms=get_valid_transforms(),
    )
    test_dataset = BirdDataset(
        Config.TEST_CSV,
        mode="test",
        load_cached_data=True,
        transforms=get_valid_transforms(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Calculate epochs based on MAX_STEPS
    steps_per_epoch = len(train_loader)
    epochs = int(Config.MAX_STEPS / max(1, steps_per_epoch))
    print(f"Training for {epochs} epochs per model (approx {Config.MAX_STEPS} steps).")

    # Storage for ensemble predictions
    val_preds_ensemble = []
    test_preds_ensemble = []

    # Ground truth for validation (extract once)
    val_targets = []
    for _, labels, _ in val_loader:
        val_targets.append(labels.numpy())
    val_targets = np.concatenate(val_targets, axis=0)

    # 3. Training Loop (Heterogeneous Ensemble)
    for backbone in Config.BACKBONES:
        print(f"\n=== Training Backbone: {backbone} ===")

        # Initialize Model
        model = BirdClassifier(backbone_name=backbone, pretrained=True)
        model.to(device)

        # Optimizer (SAM wrapping AdamW)
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(
            model.parameters(),
            base_optimizer,
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss Function
        criterion = get_weighted_loss(device)

        # Train
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device, epoch
            )

        # Evaluate on Validation
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)
        print(f"{backbone} - Val AUC: {val_auc:.6f}")

        # Generate TTA Predictions for Validation (for Ensemble)
        _, val_preds = predict_cyclic_tta(model, val_loader, device)
        val_preds_ensemble.append(val_preds)

        # Generate TTA Predictions for Test
        test_ids, test_preds = predict_cyclic_tta(model, test_loader, device)
        test_preds_ensemble.append(test_preds)

        # Cleanup to save memory
        del model, optimizer, criterion
        torch.cuda.empty_cache()

    # 4. Ensemble Aggregation
    print("\n=== Ensemble Aggregation ===")
    # Average probabilities across all models
    avg_val_preds = np.mean(val_preds_ensemble, axis=0)
    avg_test_preds = np.mean(test_preds_ensemble, axis=0)

    # Final Validation Metric
    final_auc = calculate_multilabel_auc(val_targets, avg_val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate Mean Absolute Error per sample
    errors = np.abs(val_targets - avg_val_preds)
    mean_error_per_sample = np.mean(errors, axis=1)  # (N_samples,)

    # Extract image statistics from the validation dataset
    pixel_means = []
    pixel_stds = []

    # Access images directly from the dataset (cached in memory)
    val_images = val_dataset.images

    for i in range(len(val_images)):
        img = val_images[i]
        pixel_means.append(np.mean(img))
        pixel_stds.append(np.std(img))

    pixel_means = np.array(pixel_means)
    pixel_stds = np.array(pixel_stds)

    # Compute correlations
    if len(mean_error_per_sample) == len(pixel_means):
        corr_mean, _ = pearsonr(mean_error_per_sample, pixel_means)
        corr_std, _ = pearsonr(mean_error_per_sample, pixel_stds)

        print(f"Correlation (Error vs Pixel Mean): {corr_mean:.4f}")
        print(f"Correlation (Error vs Pixel Std): {corr_std:.4f}")
    else:
        print("Error: Mismatch in sample counts for failure analysis.")

    # 6. Submission
    threshold = 0.9167709334579945
    if final_auc > threshold:
        print("\nValidation metric meets threshold. Generating submission...")

        submission_rows = []

        # test_ids contains rec_ids for the test set
        # avg_test_preds contains probs (N_test, 19)

        for idx, rec_id in enumerate(test_ids):
            probs = avg_test_preds[idx]
            for species_id, prob in enumerate(probs):
                # Construct Id as per requirement: rec_id * 100 + species_id
                row_id = int(rec_id * 100 + species_id)
                submission_rows.append({"Id": row_id, "Probability": prob})

        df_sub = pd.DataFrame(submission_rows)

        # Ensure directory exists
        os.makedirs("./submission", exist_ok=True)
        df_sub.to_csv("./submission/submission.csv", index=False)
        print("Submission saved to ./submission/submission.csv")
    else:
        print(
            f"\nValidation metric {final_auc} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
