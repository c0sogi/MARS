import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, update_bn

# Import library modules
from library.configuration import Config
from library.utilities import set_seed, get_logger, calculate_roc_auc
from library.data_loader import get_dataloaders, get_combined_dataloader
from library.architecture import get_seresnet_model
from library.training_engine import (
    train_one_epoch,
    validate,
    update_swa,
    predict_with_tta,
    save_submission,
)


def main():
    # 1. Configuration & Setup
    config = Config()
    # Cite solution_lesson_node_00044: Restore sufficient training epochs to ensure teacher convergence.
    # Under-training teachers poisons the student with poor pseudo-labels.
    config.update(
        TEACHER_EPOCHS=40,
        TEACHER_SWA_START_EPOCH=30,
        STUDENT_EPOCHS=50,
        STUDENT_SWA_START_EPOCH=35,
        BATCH_SIZE=32,
    )

    set_seed(config.SEED)
    logger = get_logger("main_pipeline")
    device = config.DEVICE

    logger.info("Starting Attentive High-Fidelity SWA-Distillation Pipeline")

    # Load DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=True
    )

    # ====================================================
    # Stage 1: Train Teacher Ensemble (SWA)
    # ====================================================
    logger.info("=== Stage 1: Training Teacher Ensemble ===")
    teacher_models = []

    for t_idx in range(config.NUM_TEACHERS):
        logger.info(f"Training Teacher {t_idx + 1}/{config.NUM_TEACHERS}")

        # Initialize Model
        model = get_seresnet_model(config).to(device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Initialize SWA Model
        swa_model = AveragedModel(model).to(device)

        # Training Loop
        for epoch in range(config.TEACHER_EPOCHS):
            loss = train_one_epoch(
                model, train_loader, optimizer, device, config, epoch
            )

            # Update SWA
            if epoch >= config.TEACHER_SWA_START_EPOCH:
                swa_model.update_parameters(model)

        # Update BN statistics for SWA model
        logger.info(f"Updating BN statistics for Teacher {t_idx + 1} SWA model...")
        update_bn(train_loader, swa_model, device=device)

        teacher_models.append(swa_model)

        # Save checkpoint (optional, but good practice)
        ckpt_path = f"{config.TEACHER_CHECKPOINT_PREFIX}_{t_idx}.pth"
        torch.save(swa_model.state_dict(), ckpt_path)

    # ====================================================
    # Stage 2: Pseudo-Labeling with TTA
    # ====================================================
    logger.info("=== Stage 2: Generating Pseudo-Labels ===")

    # Generate predictions from each teacher
    teacher_preds = []
    for t_model in teacher_models:
        preds = predict_with_tta(t_model, test_loader, device)
        teacher_preds.append(preds)

    # Average predictions (Ensemble)
    avg_preds = np.mean(teacher_preds, axis=0)

    # Sanitize (Check for NaNs)
    if np.isnan(avg_preds).any():
        logger.warning("NaNs detected in pseudo-labels. Replacing with zeros.")
        avg_preds = np.nan_to_num(avg_preds)

    # Create Pseudo-Label DataFrame
    # We need to map these back to the test metadata format
    df_test = pd.read_csv(config.TEST_METADATA_PATH)

    # Prepare dictionary for DataFrame
    pseudo_data = {
        "rec_id": df_test["rec_id"],
        "file_path": df_test["file_path"],
        "labels_str": [""] * len(df_test),  # Dummy
    }

    # Add species columns
    for i in range(config.NUM_CLASSES):
        pseudo_data[f"species_{i}"] = avg_preds[:, i]

    df_pseudo = pd.DataFrame(pseudo_data)

    # Save to Parquet
    df_pseudo.to_parquet(config.PSEUDO_LABELS_PATH, index=False)
    logger.info(f"Pseudo-labels saved to {config.PSEUDO_LABELS_PATH}")

    # ====================================================
    # Stage 3: Train Student (SWA)
    # ====================================================
    logger.info("=== Stage 3: Training Student Model ===")

    # Get Combined DataLoader
    combined_loader = get_combined_dataloader(config, config.PSEUDO_LABELS_PATH)

    # Initialize Student
    student_model = get_seresnet_model(config).to(device)
    optimizer = optim.AdamW(
        student_model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    student_swa = AveragedModel(student_model).to(device)

    # Training Loop
    for epoch in range(config.STUDENT_EPOCHS):
        loss = train_one_epoch(
            student_model, combined_loader, optimizer, device, config, epoch
        )

        if epoch >= config.STUDENT_SWA_START_EPOCH:
            student_swa.update_parameters(student_model)

    # Update BN
    logger.info("Updating BN statistics for Student SWA model...")
    update_bn(combined_loader, student_swa, device=device)

    # Save Student
    torch.save(student_swa.state_dict(), config.STUDENT_CHECKPOINT_PATH)

    # ====================================================
    # Validation & Failure Analysis
    # ====================================================
    logger.info("=== Validation & Failure Analysis ===")

    # Validate
    val_metrics = validate(student_swa, val_loader, device, config)
    final_metric = val_metrics["score"]

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Error and Image Intensity
    student_swa.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    sample_losses = []
    image_means = []

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = student_swa(images)
            loss = criterion(outputs, labels)  # (B, C)

            # Average loss per sample across classes
            loss_per_sample = loss.mean(dim=1).cpu().numpy()
            sample_losses.extend(loss_per_sample)

            # Calculate image intensity mean per sample
            # images is (B, 3, H, W). We can take mean over (1, 2, 3)
            means = images.mean(dim=[1, 2, 3]).cpu().numpy()
            image_means.extend(means)

    sample_losses = np.array(sample_losses)
    image_means = np.array(image_means)

    if len(sample_losses) > 1:
        correlation = np.corrcoef(sample_losses, image_means)[0, 1]
        print(
            f"Failure Analysis - Correlation (Error vs Image Intensity): {correlation:.4f}"
        )
    else:
        print("Failure Analysis - Not enough samples for correlation.")

    # ====================================================
    # Submission
    # ====================================================
    THRESHOLD = 0.9594082190886809

    if final_metric > THRESHOLD:
        logger.info("Metric threshold passed. Generating submission...")

        # Predict on Test Set with TTA
        test_probs = predict_with_tta(student_swa, test_loader, device)

        # Save Submission
        save_submission(test_probs, config.TEST_METADATA_PATH, config.SUBMISSION_PATH)
    else:
        logger.info(
            f"Metric {final_metric} did not exceed threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
