import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import cv2
from sklearn.metrics import roc_auc_score

# Import from provided library
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import SEResNet34
from library.engine import (
    train_one_epoch,
    valid_one_epoch,
    inference_fn,
    SWAManager,
    generate_submission,
)


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE

    print(f"Using device: {device}")

    # 2. Data Loading
    # Load initial dataloaders (Fold 0 Train, Fold 0 Val, Fold 1 Test)
    train_loader, val_loader, test_loader = get_dataloaders()

    # 3. Stage 1: Train Teacher Ensemble (SWA)
    print("\n=== Stage 1: Training Teacher Ensemble ===")
    num_teachers = 3
    teacher_models = []

    for t_idx in range(num_teachers):
        print(f"Training Teacher {t_idx + 1}/{num_teachers}...")

        # Initialize Model
        model = SEResNet34(pretrained=True).to(device)

        # Optimizer & Scheduler
        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

        # SWA Manager
        swa_manager = SWAManager(
            model,
            optimizer,
            swa_start_epoch=Config.SWA_START_EPOCH_TEACHER,
            swa_lr=Config.SWA_LR,
            device=device,
        )

        # Training Loop
        for epoch in range(Config.EPOCHS):
            # Train
            train_one_epoch(
                model, train_loader, optimizer, device, epoch, use_mixup=True
            )

            # Step Schedulers
            scheduler.step()
            swa_manager.step(epoch, model)

        # Finalize SWA (Update BatchNorm stats)
        swa_manager.update_bn(train_loader)

        # Store SWA model
        teacher_models.append(swa_manager.swa_model)

        # Save to disk
        torch.save(
            swa_manager.swa_model.state_dict(),
            os.path.join(Config.WORKING_DIR, f"teacher_{t_idx}_swa.pth"),
        )

    # 4. Stage 2: Pseudo-Labeling
    print("\n=== Stage 2: Generating Pseudo-Labels ===")

    # Inference on Test Set with Teachers + TTA
    teacher_preds = []

    for i, model in enumerate(teacher_models):
        probs, rec_ids = inference_fn(model, test_loader, device, use_tta=True)
        teacher_preds.append(probs)

    # Average predictions across ensemble
    avg_preds = np.mean(teacher_preds, axis=0)

    # Create Pseudo-labeled Dataframe
    test_df = pd.read_csv(Config.TEST_CSV)

    # Fill in predictions (test_df and test_loader are sequentially aligned)
    label_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]
    test_df[label_cols] = avg_preds

    # This dataframe now acts as extra training data
    pseudo_df = test_df

    # 5. Stage 3: Train Student (SWA)
    print("\n=== Stage 3: Training Student Model ===")

    # Get combined dataloaders (Train + Pseudo-labeled Test)
    student_train_loader, student_val_loader, student_test_loader = get_dataloaders(
        extra_train_df=pseudo_df
    )

    # Initialize Student
    student_model = SEResNet34(pretrained=True).to(device)

    # Optimizer & Scheduler
    optimizer = optim.Adam(
        student_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # SWA Manager for Student
    swa_manager_student = SWAManager(
        student_model,
        optimizer,
        swa_start_epoch=Config.SWA_START_EPOCH_STUDENT,
        swa_lr=Config.SWA_LR,
        device=device,
    )

    # Training Loop
    for epoch in range(Config.EPOCHS):
        train_one_epoch(
            student_model,
            student_train_loader,
            optimizer,
            device,
            epoch,
            use_mixup=True,
        )
        scheduler.step()
        swa_manager_student.step(epoch, student_model)

    # Finalize SWA
    swa_manager_student.update_bn(student_train_loader)
    final_model = swa_manager_student.swa_model

    # Save Student
    torch.save(
        final_model.state_dict(), os.path.join(Config.WORKING_DIR, "student_swa.pth")
    )

    # 6. Evaluation
    print("\n=== Evaluation ===")
    # Validate on Hold-out Fold 0
    val_loss, val_auc = valid_one_epoch(
        final_model, student_val_loader, device, epoch="Final"
    )

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # 7. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Get predictions on validation set
    val_probs, val_ids = inference_fn(
        final_model, student_val_loader, device, use_tta=False
    )

    # Get ground truth
    val_df = pd.read_csv(Config.VAL_CSV)
    val_targets = val_df[label_cols].values

    # Calculate Mean Absolute Error per sample
    mae_per_sample = np.mean(np.abs(val_targets - val_probs), axis=1)

    # Calculate Feature: Mean Spectrogram Intensity
    intensities = []
    for idx, row in val_df.iterrows():
        rel_path = row["file_path"]
        # Map to spectrogram
        wav_basename = os.path.basename(rel_path)
        bmp_basename = os.path.splitext(wav_basename)[0] + ".bmp"
        img_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_basename)

        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                intensities.append(np.mean(img))
            else:
                intensities.append(0.0)
        except:
            intensities.append(0.0)

    intensities = np.array(intensities)

    # Calculate Correlation
    if len(mae_per_sample) > 1:
        correlation = np.corrcoef(mae_per_sample, intensities)[0, 1]
        print(
            f"Correlation between Error Magnitude and Spectrogram Intensity: {correlation}"
        )
    else:
        print("Not enough samples for correlation analysis.")

    # 8. Submission
    SUBMISSION_THRESHOLD = 0.9594082190886809

    if val_auc > SUBMISSION_THRESHOLD:
        print("\n=== Generating Submission ===")
        # Inference on Test Set (Student Model) with TTA
        test_probs, test_ids = inference_fn(
            final_model, student_test_loader, device, use_tta=True
        )

        # Generate CSV
        generate_submission(test_probs, test_ids, Config.SUBMISSION_PATH)
    else:
        print(f"\nMetric {val_auc} <= {SUBMISSION_THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
