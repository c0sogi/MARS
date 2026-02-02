import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.dataset import CactusDataset, load_data_to_memory, get_transforms
from library.model import CactusRepVGG
from library.engine import train_model, validate, predict_tta, set_seed


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Data Loading
    print("\n[Data Loading]")

    # Load Train Data
    train_imgs, train_labels = load_data_to_memory(
        metadata_path=Config.TRAIN_META_PATH,
        cache_imgs_path=Config.CACHE_TRAIN_IMGS,
        cache_labels_path=Config.CACHE_TRAIN_LABELS,
        load_cached_data=True,
        is_test=False,
    )

    # Load Validation Data
    val_imgs, val_labels = load_data_to_memory(
        metadata_path=Config.VAL_META_PATH,
        cache_imgs_path=Config.CACHE_VAL_IMGS,
        cache_labels_path=Config.CACHE_VAL_LABELS,
        load_cached_data=True,
        is_test=False,
    )

    # Load Test Data
    test_imgs, test_ids = load_data_to_memory(
        metadata_path=Config.TEST_META_PATH,
        cache_imgs_path=Config.CACHE_TEST_IMGS,
        cache_ids_path=Config.CACHE_TEST_IDS,
        load_cached_data=True,
        is_test=True,
    )

    # Create Datasets
    train_dataset = CactusDataset(
        train_imgs, labels=train_labels, transform=get_transforms(split="train")
    )
    val_dataset = CactusDataset(
        val_imgs, labels=val_labels, transform=get_transforms(split="val")
    )
    test_dataset = CactusDataset(
        test_imgs, ids=test_ids, transform=get_transforms(split="test")
    )

    # Create DataLoaders
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

    # 3. Model Training
    print("\n[Model Training]")
    model = CactusRepVGG(num_classes=Config.NUM_CLASSES, deploy=False)
    model = model.to(device)

    # Train the model (engine handles loop, scheduler, early stopping)
    best_auc_score = train_model(model, train_loader, val_loader, device)

    # 4. Final Validation & Failure Analysis
    print("\n[Validation & Failure Analysis]")

    # Load best model state
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH))
    model.eval()

    # Calculate Final Metric
    criterion = nn.BCEWithLogitsLoss()
    val_loss, final_auc = validate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis: Correlation between Error and Input Features
    # Get predictions for validation set (without TTA for direct analysis)
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)  # Returns main_out in eval mode
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_preds.extend(probs)
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate Error Magnitude
    errors = np.abs(all_targets - all_preds)

    # Calculate Image Features (Mean Intensity, Contrast) from loaded numpy arrays
    # val_imgs is (N, 32, 32, 3)
    img_means = val_imgs.mean(axis=(1, 2, 3))
    img_contrasts = val_imgs.std(axis=(1, 2, 3))

    # Calculate Correlations
    corr_mean, _ = pearsonr(errors, img_means)
    corr_contrast, _ = pearsonr(errors, img_contrasts)

    print("\nFailure Analysis (Correlation with Error Magnitude):")
    print(f"  Image Mean Intensity: {corr_mean:.4f}")
    print(f"  Image Contrast:       {corr_contrast:.4f}")

    # 5. Submission
    # The prompt condition "metric > 1.0" is mathematically impossible for AUC.
    # Assuming standard threshold of 0.5 or simply "if valid".
    if final_auc > 0.5:
        print("\n[Submission Generation]")

        # Switch model to deploy mode (fuses layers, removes aux head)
        model.switch_to_deploy()

        # Generate predictions with TTA
        predictions_map = predict_tta(model, test_loader, device)

        # Create DataFrame
        submission_df = pd.DataFrame(
            list(predictions_map.items()), columns=["id", "has_cactus"]
        )

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(submission_df.head())
    else:
        print(f"Validation AUC ({final_auc}) too low. Skipping submission.")


if __name__ == "__main__":
    main()
