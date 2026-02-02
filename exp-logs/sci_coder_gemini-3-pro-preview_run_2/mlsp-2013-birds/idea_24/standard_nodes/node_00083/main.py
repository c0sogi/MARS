import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import set_seed, calculate_roc_auc, save_submission
from library.dataset import load_data, BirdDataset, MixupCollate
from library.models import BirdClassifier
from library.engine import train_fn, eval_fn, inference_fn
from library.sam import SAM


def main():
    # 1. Setup & Configuration
    set_seed(Config.SEED)
    device = Config.DEVICE

    # 2. Data Loading
    # Load training and validation data using the provided metadata
    # load_cached_data=True to use preprocessed .npy files if available
    train_images, train_labels, train_ids = load_data(
        Config.TRAIN_CSV, "train", load_cached_data=True
    )
    val_images, val_labels, val_ids = load_data(
        Config.VAL_CSV, "val", load_cached_data=True
    )

    # Calculate positive weights for BCEWithLogitsLoss to handle class imbalance
    # pos_weight = (num_neg / num_pos)
    # Add epsilon to avoid division by zero
    num_pos = np.sum(train_labels, axis=0)
    num_neg = len(train_labels) - num_pos
    pos_weights = torch.tensor((num_neg / (num_pos + 1e-6)), dtype=torch.float32).to(
        device
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # Create Datasets
    train_dataset = BirdDataset(train_images, train_labels, train_ids, split="train")
    val_dataset = BirdDataset(val_images, val_labels, val_ids, split="val")

    # Create DataLoaders
    # Use MixupCollate for training
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=MixupCollate(alpha=Config.MIXUP_ALPHA),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Training (Heterogeneous Ensemble)
    # We train one model per architecture defined in Config.BACKBONES
    trained_models = []

    # Calculate epochs based on target total steps
    steps_per_epoch = len(train_loader)
    num_epochs = int(np.ceil(Config.TOTAL_STEPS / steps_per_epoch))

    # Limit epochs to a reasonable number if dataset is very small to avoid overfitting/waste
    # But ensure at least some training happens.
    num_epochs = min(num_epochs, Config.MAX_EPOCHS)

    print(
        f"Training Config: {len(Config.BACKBONES)} Backbones, {num_epochs} Epochs each."
    )

    for backbone_name in Config.BACKBONES:
        # Initialize Model
        model = BirdClassifier(backbone_name).to(device)

        # Initialize SAM Optimizer
        # Base optimizer is AdamW
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(
            model.parameters(),
            base_optimizer,
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
            rho=0.05,
        )

        # Scheduler: Constant (as per strategy)
        scheduler = None

        best_val_auc = 0.0
        best_model_state = None

        for epoch in range(num_epochs):
            # Train
            train_loss = train_fn(
                model, train_loader, optimizer, device, scheduler, criterion
            )

            # Validate
            val_loss, val_auc, _, _ = eval_fn(model, val_loader, device, criterion)

            # Save best model state
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict()

        # Load best state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        trained_models.append(model)
        # Clear memory
        torch.cuda.empty_cache()

    # 4. Ensemble Validation & Failure Analysis
    print("Performing Ensemble Validation...")

    ensemble_preds = np.zeros((len(val_labels), Config.NUM_CLASSES), dtype=np.float32)

    for model in trained_models:
        _, _, preds, _ = eval_fn(model, val_loader, device, criterion)
        ensemble_preds += preds

    # Average predictions
    ensemble_preds /= len(trained_models)

    # Final Validation Metric
    final_val_auc = calculate_roc_auc(val_labels, ensemble_preds)
    print(f"Final Validation Metric: {final_val_auc}")

    # Failure Analysis
    # Correlate error magnitude with input feature (Pixel Mean Intensity)
    # We calculate the mean error per sample (averaged across classes)
    abs_error = np.abs(val_labels - ensemble_preds).mean(axis=1)

    # Calculate mean pixel intensity for each validation image
    # val_images is (N, H, W, 3) or (N, H, W)
    if val_images.ndim == 4:
        pixel_means = val_images.mean(axis=(1, 2, 3))
    else:
        pixel_means = val_images.mean(axis=(1, 2))

    # Calculate correlation
    corr, _ = pearsonr(abs_error, pixel_means)
    print(f"Failure Analysis - Correlation (Pixel Mean vs Error): {corr:.4f}")

    # 5. Submission
    threshold = 0.9167709334579945

    if final_val_auc > threshold:
        # Load Test Data
        test_images, test_labels, test_ids = load_data(
            Config.TEST_CSV, "test", load_cached_data=True
        )

        final_test_preds = np.zeros(
            (len(test_images), Config.NUM_CLASSES), dtype=np.float32
        )

        # Inference with Cyclic TTA for each model
        for model in trained_models:
            _, preds = inference_fn(model, test_images, test_ids, device)
            final_test_preds += preds

        # Average across models
        final_test_preds /= len(trained_models)

        # Save Submission
        submission_path = "./submission/submission.csv"
        save_submission(test_ids, final_test_preds, submission_path)


if __name__ == "__main__":
    main()
