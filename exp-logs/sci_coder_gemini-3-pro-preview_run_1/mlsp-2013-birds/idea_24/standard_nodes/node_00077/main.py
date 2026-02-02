import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config
from library.utils import set_seed, get_logger
from library.dataset import load_data, BirdDataset, get_transforms
from library.model import create_model
from library.training import train_model, validate
from library.inference import predict_ensemble, predict_test_set, create_submission


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE

    # Create working directory if not exists (already handled in Config, but good practice)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Logger
    logger = get_logger(os.path.join(Config.WORKING_DIR, "run.log"))
    logger.info(
        "Starting Diversity-Augmented ResNet34-d Ensemble Distillation Pipeline"
    )

    # 2. Data Loading
    # Load metadata (Train: Fold 0 split, Val: Fold 0 split, Test: Fold 1)
    train_df, val_df, test_df = load_data(load_cached_data=True)

    # Create Datasets
    train_dataset = BirdDataset(
        train_df, transforms=get_transforms("train"), mode="train"
    )
    val_dataset = BirdDataset(val_df, transforms=get_transforms("val"), mode="val")
    # Test dataset for pseudo-labeling and final submission
    test_dataset = BirdDataset(test_df, transforms=get_transforms("test"), mode="test")

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

    # =========================================================================
    # Stage 1: Diversity-Augmented Teacher Ensemble Training
    # =========================================================================
    logger.info("Stage 1: Training Teacher Ensemble")

    teachers = []

    for i in range(Config.NUM_TEACHERS):
        teacher_idx = i + 1
        mixup_alpha = Config.TEACHER_MIXUP_ALPHAS[i]
        model_alias = f"teacher_{teacher_idx}"

        logger.info(
            f"Training Teacher {teacher_idx}/{Config.NUM_TEACHERS} with Mixup Alpha {mixup_alpha}"
        )

        # Initialize Model
        model = create_model(pretrained=Config.PRETRAINED)
        model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Train
        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            mixup_alpha=mixup_alpha,
            swa_start_epoch=Config.TEACHER_SWA_START_EPOCH,
            swa_lr=Config.TEACHER_SWA_LR,
            epochs=Config.EPOCHS,
            save_dir=Config.WORKING_DIR,
            model_alias=model_alias,
        )

        teachers.append(trained_model)

    # =========================================================================
    # Stage 2: Robust Pseudo-Label Generation
    # =========================================================================
    logger.info("Stage 2: Generating Pseudo-Labels for Test Set")

    # Predict on Test Set using Ensemble
    test_ids, test_probs = predict_ensemble(
        models=teachers, loader=test_loader, device=device, tta=Config.TTA_FLIP
    )

    # Create Pseudo-Labeled DataFrame
    # We need to structure it exactly like train_df
    pseudo_df = test_df.copy()

    # Update species columns with predicted probabilities (soft labels)
    # The Dataset class handles float labels correctly
    for idx, col in enumerate([f"species_{k}" for k in range(Config.NUM_CLASSES)]):
        pseudo_df[col] = test_probs[:, idx]

    # Combine Train and Pseudo-Labeled Test
    combined_df = pd.concat([train_df, pseudo_df], axis=0).reset_index(drop=True)
    logger.info(
        f"Combined Dataset Size: {len(combined_df)} (Train: {len(train_df)} + Pseudo: {len(pseudo_df)})"
    )

    # Create Combined DataLoader
    combined_dataset = BirdDataset(
        combined_df, transforms=get_transforms("train"), mode="train"
    )
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # =========================================================================
    # Stage 3: Student Training
    # =========================================================================
    logger.info("Stage 3: Training Student Model")

    student_model = create_model(pretrained=Config.PRETRAINED)
    student_model.to(device)

    student_optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Train Student
    final_student = train_model(
        model=student_model,
        train_loader=combined_loader,
        val_loader=val_loader,  # Validate on original clean validation set
        optimizer=student_optimizer,
        device=device,
        mixup_alpha=Config.STUDENT_MIXUP_ALPHA,
        swa_start_epoch=Config.STUDENT_SWA_START_EPOCH,
        swa_lr=Config.STUDENT_SWA_LR,
        epochs=Config.EPOCHS,
        save_dir=Config.WORKING_DIR,
        model_alias="student",
    )

    # =========================================================================
    # Validation & Failure Analysis
    # =========================================================================
    logger.info("Performing Final Validation and Failure Analysis")

    # 1. Final Validation Metric
    criterion = torch.nn.BCEWithLogitsLoss()
    val_loss, val_auc = validate(final_student, val_loader, criterion, device)

    print(f"Final Validation Metric: {val_auc}")

    # 2. Failure Analysis
    # We need predictions and targets
    final_student.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = final_student(images)
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate Mean Absolute Error per sample
    # Shape: (N_samples, N_classes)
    abs_errors = np.abs(all_preds - all_targets)
    mean_abs_error_per_sample = np.mean(abs_errors, axis=1)

    # Feature for correlation: Number of labels (Species Count)
    label_counts = np.sum(all_targets, axis=1)

    # Calculate Correlation
    if np.std(label_counts) > 0 and np.std(mean_abs_error_per_sample) > 0:
        correlation = np.corrcoef(label_counts, mean_abs_error_per_sample)[0, 1]
    else:
        correlation = 0.0

    print(f"Correlation between Error Magnitude and Species Count: {correlation:.4f}")

    # =========================================================================
    # Submission
    # =========================================================================
    THRESHOLD = 0.9594082190886809

    if val_auc > THRESHOLD:
        logger.info(
            f"Validation AUC ({val_auc}) exceeds threshold ({THRESHOLD}). Generating submission."
        )

        # Predict on Test Set with Student Model
        # Using TTA as per config
        sub_ids, sub_probs = predict_test_set(
            final_student, test_loader, device, tta=Config.TTA_FLIP
        )

        create_submission(sub_ids, sub_probs, Config.SUBMISSION_PATH)
    else:
        logger.info(
            f"Validation AUC ({val_auc}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
