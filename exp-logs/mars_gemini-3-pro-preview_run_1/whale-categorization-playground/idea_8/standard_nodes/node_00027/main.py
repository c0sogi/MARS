import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import provided libraries
from library import config
from library import utils
from library import transforms
from library import dataset
from library import model as model_lib
from library import loss as loss_lib
from library import engine

# Configuration for this run
BATCH_SIZE = config.BATCH_SIZE
NUM_WORKERS = 2


def run_training():
    """
    Trains an ensemble of 5 DenseNet121 models using different seeds.
    """
    # 1. Setup Data
    # Load training data
    train_dataset = dataset.WhaleDataset(
        csv_file=config.TRAIN_CSV,
        img_dir=config.TRAIN_IMG_DIR,
        transform=transforms.get_train_transforms(),
        load_cached_data=True,
    )

    # We need the class mapping to ensure consistency across models and inference
    class_mapping = train_dataset.classes
    num_classes = len(class_mapping)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    models = []

    # 2. Train Ensemble
    for seed_idx, seed in enumerate(config.SEEDS):
        # Set seed for reproducibility for this specific model run
        utils.seed_everything(seed)

        print(f"Training Model {seed_idx+1}/{len(config.SEEDS)} (Seed {seed})...")

        # Initialize Model
        net = model_lib.WhaleDenseNet(
            num_classes=num_classes, embedding_dim=config.EMBEDDING_DIM, pretrained=True
        )
        net = net.to(config.DEVICE)

        # Optimizer & Scheduler
        optimizer = optim.AdamW(
            net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.EPOCHS, eta_min=config.SCHEDULER_MIN_LR
        )

        # Loss Function
        criterion = loss_lib.LabelSmoothingCrossEntropy(
            smoothing=config.LABEL_SMOOTHING
        )

        # Training Loop
        for epoch in range(config.EPOCHS):
            avg_loss = engine.train_one_epoch(
                net, train_loader, criterion, optimizer, config.DEVICE
            )
            scheduler.step()

        # Move model to CPU to free up GPU memory for the next training iteration
        net.eval()
        net.cpu()
        models.append(net)

        # Clean up to prevent memory leaks
        del optimizer, scheduler, criterion
        torch.cuda.empty_cache()

    return models, class_mapping


def validate_ensemble(models, class_mapping):
    """
    Validates the ensemble on the hold-out set and performs failure analysis.
    """
    print("Starting Ensemble Validation...")

    val_dataset = dataset.WhaleDataset(
        csv_file=config.VAL_CSV,
        img_dir=config.TRAIN_IMG_DIR,  # Validation images are in the train folder
        transform=transforms.get_test_transforms(),
        class_mapping=class_mapping,
        load_cached_data=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    all_targets = []
    all_preds = []

    # For Failure Analysis
    error_magnitudes = []  # 1.0 if correct class not in top 5, else 0.0
    feature_intensities = []  # Mean pixel intensity of the image

    # Move models to GPU for inference
    for model in models:
        model.to(config.DEVICE)
        model.eval()

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(config.DEVICE)

            # Ensemble Accumulator
            ensemble_logits = None

            for model in models:
                # TTA View 1: Original
                logits_1 = model(images, labels=None)
                # TTA View 2: Horizontal Flip
                logits_2 = model(torch.flip(images, dims=[3]), labels=None)

                avg_logits = (logits_1 + logits_2) / 2.0

                if ensemble_logits is None:
                    ensemble_logits = avg_logits
                else:
                    ensemble_logits += avg_logits

            # Average over models
            ensemble_logits /= len(models)

            # Get Top 5 Predictions
            _, top_indices = torch.topk(ensemble_logits, k=5, dim=1)

            # Convert to numpy
            batch_preds = top_indices.cpu().numpy()
            batch_targets = labels.numpy()

            all_preds.extend(batch_preds)
            all_targets.extend(batch_targets)

            # Failure Analysis Data Collection
            # Calculate mean intensity of images (normalized)
            # images shape is (B, 3, H, W) -> mean over dims (1, 2, 3)
            means = images.mean(dim=(1, 2, 3)).cpu().numpy()
            feature_intensities.extend(means)

            for i in range(len(batch_targets)):
                target = batch_targets[i]
                preds = batch_preds[i]

                # Define Error: 1 if target NOT in preds, 0 if it is (Hit)
                is_error = 1.0 if target not in preds else 0.0
                error_magnitudes.append(is_error)

    # Move models back to CPU to free memory
    for model in models:
        model.cpu()
    torch.cuda.empty_cache()

    # Compute MAP@5 Metric
    score = utils.map5(all_targets, all_preds)
    print(f"Final Validation Metric: {score}")

    # Failure Analysis Calculation
    if len(error_magnitudes) > 1:
        # Calculate Point-Biserial Correlation between Binary Error and Continuous Intensity
        corr, _ = pearsonr(error_magnitudes, feature_intensities)
        print(f"Failure Analysis - Correlation (Error vs Intensity): {corr:.4f}")
    else:
        print("Failure Analysis - Not enough samples for correlation.")

    return score


def generate_submission(models, class_mapping):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating Submission...")

    test_dataset = dataset.WhaleDataset(
        csv_file=config.TEST_CSV,
        img_dir=config.TEST_IMG_DIR,
        transform=transforms.get_test_transforms(),
        class_mapping=class_mapping,  # Use same mapping as training
        load_cached_data=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Move models to GPU
    for model in models:
        model.to(config.DEVICE)
        model.eval()

    all_image_ids = []
    all_top_labels = []

    # Inverse mapping: Index -> String ID
    idx_to_label = {v: k for k, v in test_dataset.label_to_idx.items()}

    with torch.no_grad():
        for batch in test_loader:
            images, img_names = batch
            images = images.to(config.DEVICE)

            ensemble_logits = None

            for model in models:
                # TTA View 1: Original
                logits_1 = model(images, labels=None)
                # TTA View 2: Horizontal Flip
                logits_2 = model(torch.flip(images, dims=[3]), labels=None)

                avg_logits = (logits_1 + logits_2) / 2.0

                if ensemble_logits is None:
                    ensemble_logits = avg_logits
                else:
                    ensemble_logits += avg_logits

            ensemble_logits /= len(models)

            # Get Top 5
            _, top_indices = torch.topk(ensemble_logits, k=5, dim=1)

            top_indices = top_indices.cpu().numpy()

            all_image_ids.extend(img_names)

            # Convert indices to space-separated strings
            for row_indices in top_indices:
                labels_str = [idx_to_label[idx] for idx in row_indices]
                all_top_labels.append(" ".join(labels_str))

    # Create DataFrame
    df_sub = pd.DataFrame({"Image": all_image_ids, "Id": all_top_labels})

    # Save
    sub_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    df_sub.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")


if __name__ == "__main__":
    # 1. Train Ensemble
    trained_models, class_map = run_training()

    # 2. Validate Ensemble
    val_score = validate_ensemble(trained_models, class_map)

    # 3. Generate Submission if Threshold Met
    THRESHOLD = 0.6545824094604581
    if val_score > THRESHOLD:
        generate_submission(trained_models, class_map)
    else:
        print(
            f"Validation score {val_score} did not meet threshold {THRESHOLD}. Skipping submission."
        )
