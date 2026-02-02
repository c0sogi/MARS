import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr
import gc

# Import from library
from library.config import Config
from library.dataset import PathologyDataset, get_transforms
from library.model import EnsembleDenseNet
from library.engine import train_one_epoch, validate, predict_tta
from library.utils import seed_everything, EarlyStopping, calculate_auc


def main():
    # 1. Setup
    # Override Config for Fast Baseline execution
    Config.EPOCHS = 5
    Config.BATCH_SIZE = 256

    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Running on device: {device}")
    print(f"Training with {Config.EPOCHS} epochs per model.")

    # 2. Data Preparation
    # Load metadata
    # We combine train and val metadata to perform 5-fold CV on the full dataset
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    full_df = pd.concat([train_meta, val_meta], ignore_index=True)

    # Create Stratified Folds
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )
    full_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(full_df, full_df["label"])):
        full_df.loc[val_idx, "fold"] = fold

    print(f"Total samples for CV: {len(full_df)}")

    # Placeholders for OOF predictions
    # We will accumulate predictions from each architecture
    # oof_preds will store the sum of probabilities, which we will average later
    oof_preds_sum = np.zeros(len(full_df))
    oof_targets = full_df["label"].values

    # Store paths of saved models for final inference
    model_paths = []

    # 3. Training Loop
    for arch in Config.ARCHITECTURES:
        print(f"\n{'='*30}\nTraining Architecture: {arch}\n{'='*30}")

        # Store predictions for this specific architecture
        arch_oof_preds = np.zeros(len(full_df))

        for fold in range(Config.NUM_FOLDS):
            print(f"\n--- Fold {fold} ---")

            # Split Data
            train_df = full_df[full_df["fold"] != fold].reset_index(drop=True)
            val_df = full_df[full_df["fold"] == fold].reset_index(drop=True)
            val_indices = full_df[full_df["fold"] == fold].index.values

            # Datasets & Loaders
            train_dataset = PathologyDataset(
                train_df, transforms=get_transforms("train")
            )
            val_dataset = PathologyDataset(val_df, transforms=get_transforms("val"))

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

            # Initialize Model
            model = EnsembleDenseNet(
                arch_name=arch, pretrained=True, num_classes=Config.NUM_CLASSES
            )
            model = model.to(device)

            # Optimizer & Scheduler
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.EPOCHS
            )

            # Early Stopping
            checkpoint_path = os.path.join(Config.WORKING_DIR, f"{arch}_fold{fold}.pth")
            early_stopping = EarlyStopping(
                patience=Config.PATIENCE, verbose=False, path=checkpoint_path
            )

            # Training Epochs
            for epoch in range(Config.EPOCHS):
                train_loss = train_one_epoch(
                    epoch, model, train_loader, optimizer, device, scheduler
                )
                val_loss, val_auc = validate(model, val_loader, device)

                early_stopping(val_auc, model, optimizer, epoch)

                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break

            # Load Best Model for OOF Inference
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            model_paths.append(checkpoint_path)

            # Generate OOF Preds for this fold
            model.eval()
            fold_preds = []
            with torch.no_grad():
                for images, _, _ in val_loader:
                    images = images.to(device)
                    outputs = model(images)
                    preds = torch.sigmoid(outputs.view(-1))
                    fold_preds.extend(preds.cpu().numpy())

            arch_oof_preds[val_indices] = np.array(fold_preds)

            # Cleanup to free memory
            del (
                model,
                optimizer,
                scheduler,
                train_loader,
                val_loader,
                train_dataset,
                val_dataset,
            )
            torch.cuda.empty_cache()
            gc.collect()

        # Add current architecture's predictions to the ensemble sum
        oof_preds_sum += arch_oof_preds

    # Average predictions across architectures
    final_oof_preds = oof_preds_sum / len(Config.ARCHITECTURES)

    # 4. Validation Assessment
    final_auc = calculate_auc(oof_targets, final_oof_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate absolute errors
    errors = np.abs(oof_targets - final_oof_preds)

    # Sample a subset for feature computation to save time
    sample_size = min(2000, len(full_df))
    analysis_indices = np.random.choice(len(full_df), size=sample_size, replace=False)

    brightness_vals = []
    contrast_vals = []
    sampled_errors = errors[analysis_indices]

    # Create dataset for analysis
    analysis_df = full_df.iloc[analysis_indices].reset_index(drop=True)
    # Use 'val' transforms to get deterministic normalized images
    analysis_dataset = PathologyDataset(analysis_df, transforms=get_transforms("val"))
    analysis_loader = DataLoader(
        analysis_dataset, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    for images, _, _ in analysis_loader:
        # Images are (B, C, H, W)
        # Compute stats per image
        # Brightness: Mean across spatial dims and channels
        b = images.mean(dim=(1, 2, 3)).numpy()
        # Contrast: Std across spatial dims and channels
        c = images.std(dim=(1, 2, 3)).numpy()

        brightness_vals.extend(b)
        contrast_vals.extend(c)

    brightness_vals = np.array(brightness_vals)
    contrast_vals = np.array(contrast_vals)

    # Calculate correlations
    corr_b, _ = pearsonr(sampled_errors, brightness_vals)
    corr_c, _ = pearsonr(sampled_errors, contrast_vals)

    print(f"Correlation between Error and Brightness: {corr_b}")
    print(f"Correlation between Error and Contrast: {corr_c}")

    # 6. Submission
    THRESHOLD = 0.9849192531860572

    if final_auc > THRESHOLD:
        print("\nValidation metric meets threshold. Generating submission...")

        # Load Test Data
        test_df = pd.read_csv(Config.TEST_METADATA_PATH)
        test_dataset = PathologyDataset(test_df, transforms=get_transforms("test"))
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Ensemble Inference
        final_test_preds = np.zeros(len(test_df))

        for mp in model_paths:
            # Determine architecture from filename
            arch_name = "densenet121" if "densenet121" in mp else "densenet169"

            model = EnsembleDenseNet(
                arch_name=arch_name, pretrained=False, num_classes=Config.NUM_CLASSES
            )
            checkpoint = torch.load(mp)
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device)

            # TTA Inference
            df_preds = predict_tta(model, test_loader, device)
            final_test_preds += df_preds["label"].values

            del model
            torch.cuda.empty_cache()
            gc.collect()

        # Average across all models
        final_test_preds /= len(model_paths)

        # Create Submission File
        submission = pd.DataFrame({"id": test_df["id"], "label": final_test_preds})

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric {final_auc} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
